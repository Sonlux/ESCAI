"""
Example demonstrating the Performance Prediction system in the ESCAI framework.

This example shows how to:
1. Create epistemic states and agent states
2. Predict task success probability
3. Estimate completion time
4. Identify risk factors
5. Get intervention recommendations
"""

import asyncio
from datetime import datetime, timedelta

from escai_framework.core.performance_predictor import (
    PerformancePredictor, ExecutionStep, AgentState
)
from escai_framework.models.epistemic_state import (
    EpistemicState, BeliefState, KnowledgeState, GoalState,
    BeliefType, GoalStatus
)
from escai_framework.models.prediction_result import PredictionType


async def main():
    """Demonstrate performance prediction capabilities."""
    print("🔮 ESCAI Performance Prediction System Demo")
    print("=" * 50)
    
    # Initialize the performance predictor
    predictor = PerformancePredictor()
    
    # Create sample epistemic state
    belief_states = [
        BeliefState(
            content="The API endpoint is reliable",
            belief_type=BeliefType.PROBABILISTIC,
            confidence=0.8,
            evidence=["Previous successful calls", "Monitoring data"]
        ),
        BeliefState(
            content="Database connection is stable",
            belief_type=BeliefType.FACTUAL,
            confidence=0.9,
            evidence=["Connection test passed"]
        ),
        BeliefState(
            content="Processing might take longer than expected",
            belief_type=BeliefType.PROBABILISTIC,
            confidence=0.6,
            evidence=["Large dataset size", "Complex transformations"]
        )
    ]
    
    knowledge_state = KnowledgeState(
        facts=[
            "API rate limit is 1000 requests/hour",
            "Database has 1M records",
            "Processing requires 3 transformation steps"
        ],
        rules=[
            "If API fails, retry with exponential backoff",
            "If memory usage > 80%, trigger garbage collection"
        ],
        concepts={
            "API": {"type": "service", "reliability": 0.95},
            "Database": {"type": "storage", "size": "large"},
            "Processing": {"type": "computation", "complexity": "high"}
        },
        relationships=[
            {"subject": "API", "predicate": "depends_on", "object": "Database"},
            {"subject": "Processing", "predicate": "uses", "object": "API"}
        ],
        confidence_score=0.8
    )
    
    goal_states = [
        GoalState(
            description="Process all customer data",
            status=GoalStatus.ACTIVE,
            priority=9,
            progress=0.3
        ),
        GoalState(
            description="Generate analytics report",
            status=GoalStatus.ACTIVE,
            priority=7,
            progress=0.1
        ),
        GoalState(
            description="Validate data quality",
            status=GoalStatus.ACTIVE,
            priority=8,
            progress=0.5
        )
    ]
    
    epistemic_state = EpistemicState(
        agent_id="data_processing_agent",
        timestamp=datetime.utcnow(),
        belief_states=belief_states,
        knowledge_state=knowledge_state,
        goal_states=goal_states,
        confidence_level=0.75,
        uncertainty_score=0.25,
        decision_context={
            "task_type": "data_processing",
            "complexity": "high",
            "dataset_size": "1M_records",
            "time_constraint": "4_hours"
        }
    )
    
    # Create execution history
    execution_history = []
    base_time = datetime.utcnow() - timedelta(minutes=30)
    
    steps_data = [
        ("Initialize connection", 2000, True),
        ("Load configuration", 1500, True),
        ("Validate input data", 5000, True),
        ("Start data processing", 3000, True),
        ("Process batch 1", 45000, True),
        ("Process batch 2", 48000, True),
        ("Process batch 3", 52000, False),  # This step failed
        ("Retry batch 3", 55000, True),
        ("Process batch 4", 47000, True)
    ]
    
    for i, (action, duration, success) in enumerate(steps_data):
        step = ExecutionStep(
            step_id=f"step_{i+1}",
            timestamp=base_time + timedelta(seconds=i*60),
            action=action,
            duration_ms=duration,
            success=success,
            epistemic_state=epistemic_state if i % 3 == 0 else None,  # Include epistemic state for some steps
            context={
                "batch_number": i+1 if "batch" in action else None,
                "retry_count": 1 if "retry" in action.lower() else 0,
                "memory_usage": 0.4 + (i * 0.05),  # Increasing memory usage
                "cpu_usage": 0.3 + (i * 0.03)
            }
        )
        execution_history.append(step)
    
    # Create agent state
    agent_state = AgentState(
        agent_id="data_processing_agent",
        current_task="process_customer_data_analytics",
        execution_history=execution_history,
        epistemic_state=epistemic_state,
        resource_usage={
            "cpu": 0.65,
            "memory": 0.72,
            "network": 0.45,
            "disk_io": 0.38
        },
        performance_metrics={
            "accuracy": 0.94,
            "throughput": 0.78,
            "efficiency": 0.82,
            "error_rate": 0.06
        }
    )
    
    print("\n📊 Agent State Summary:")
    print(f"Agent ID: {agent_state.agent_id}")
    print(f"Current Task: {agent_state.current_task}")
    print(f"Execution Steps: {len(agent_state.execution_history)}")
    print(f"Confidence Level: {epistemic_state.confidence_level:.2f}")
    print(f"Uncertainty Score: {epistemic_state.uncertainty_score:.2f}")
    print(f"Resource Usage: CPU={agent_state.resource_usage['cpu']:.1%}, Memory={agent_state.resource_usage['memory']:.1%}")
    
    # 1. Predict Success Probability
    print("\n🎯 Success Probability Prediction:")
    print("-" * 30)
    
    success_prediction = await predictor.predict_success(epistemic_state)
    
    print(f"Predicted Success Probability: {success_prediction.predicted_value:.1%}")
    print(f"Prediction Confidence: {success_prediction.confidence_score:.1%}")
    print(f"Risk Level: {success_prediction.risk_level.value.title()}")
    
    if success_prediction.confidence_interval:
        ci = success_prediction.confidence_interval
        print(f"Confidence Interval: [{ci.lower_bound:.1%}, {ci.upper_bound:.1%}] ({ci.confidence_level:.0%} confidence)")
    
    print(f"Model Used: {success_prediction.model_name}")
    print(f"Features Analyzed: {len(success_prediction.features_used)}")
    
    # 2. Estimate Completion Time
    print("\n⏱️ Completion Time Estimation:")
    print("-" * 30)
    
    time_estimate = await predictor.estimate_completion_time(execution_history)
    
    estimated_minutes = time_estimate.estimated_duration_ms / (1000 * 60)
    print(f"Estimated Completion Time: {estimated_minutes:.1f} minutes")
    
    ci = time_estimate.confidence_interval
    ci_min_lower = ci.lower_bound / (1000 * 60)
    ci_min_upper = ci.upper_bound / (1000 * 60)
    print(f"Time Range: {ci_min_lower:.1f} - {ci_min_upper:.1f} minutes ({ci.confidence_level:.0%} confidence)")
    
    print(f"Factors Considered: {', '.join(time_estimate.factors_considered)}")
    
    # 3. Identify Risk Factors
    print("\n⚠️ Risk Factor Analysis:")
    print("-" * 30)
    
    risk_factors = await predictor.identify_risk_factors(agent_state)
    
    if risk_factors:
        print(f"Identified {len(risk_factors)} risk factors:")
        for i, risk in enumerate(risk_factors, 1):
            risk_score = risk.calculate_risk_score()
            print(f"\n{i}. {risk.name}")
            print(f"   Category: {risk.category.title()}")
            print(f"   Risk Score: {risk_score:.2f} (Impact: {risk.impact_score:.2f}, Probability: {risk.probability:.2f})")
            print(f"   Description: {risk.description}")
            if risk.mitigation_strategies:
                print(f"   Mitigation: {risk.mitigation_strategies[0]}")
    else:
        print("No significant risk factors identified.")
    
    # 4. Get Intervention Recommendations
    print("\n💡 Intervention Recommendations:")
    print("-" * 30)
    
    interventions = await predictor.recommend_interventions(success_prediction)
    
    if interventions:
        print(f"Recommended {len(interventions)} interventions:")
        for i, intervention in enumerate(interventions, 1):
            benefit_cost = intervention.calculate_benefit_cost_ratio()
            print(f"\n{i}. {intervention.name}")
            print(f"   Type: {intervention.intervention_type.value.replace('_', ' ').title()}")
            print(f"   Expected Impact: {intervention.expected_impact:.1%}")
            print(f"   Implementation Cost: {intervention.implementation_cost:.1%}")
            print(f"   Urgency: {intervention.urgency.value.title()}")
            print(f"   Benefit/Cost Ratio: {benefit_cost:.2f}")
            print(f"   Description: {intervention.description}")
    else:
        print("No interventions recommended at this time.")
    
    # 5. Performance Summary
    print("\n📈 Performance Summary:")
    print("-" * 30)
    
    # Calculate some derived metrics
    successful_steps = sum(1 for step in execution_history if step.success)
    success_rate = successful_steps / len(execution_history)
    avg_duration = sum(step.duration_ms for step in execution_history) / len(execution_history)
    
    print(f"Historical Success Rate: {success_rate:.1%}")
    print(f"Average Step Duration: {avg_duration/1000:.1f} seconds")
    print(f"Current Goal Progress: {sum(g.progress for g in goal_states)/len(goal_states):.1%}")
    
    # Risk assessment
    overall_risk = success_prediction.calculate_overall_risk_score()
    print(f"Overall Risk Score: {overall_risk:.2f}")
    
    if overall_risk < 0.3:
        risk_assessment = "Low Risk - Proceeding as planned"
    elif overall_risk < 0.6:
        risk_assessment = "Medium Risk - Monitor closely"
    else:
        risk_assessment = "High Risk - Consider interventions"
    
    print(f"Risk Assessment: {risk_assessment}")
    
    # Recommendations
    print(f"\n🎯 Key Recommendations:")
    if success_prediction.predicted_value > 0.8:
        print("✅ High success probability - continue current approach")
    elif success_prediction.predicted_value > 0.6:
        print("⚠️ Moderate success probability - consider optimizations")
    else:
        print("🚨 Low success probability - immediate intervention recommended")
    
    if estimated_minutes > 120:  # More than 2 hours
        print("⏰ Long completion time estimated - consider resource scaling")
    
    if agent_state.resource_usage['memory'] > 0.7:
        print("💾 High memory usage detected - monitor for potential issues")
    
    print("\n" + "=" * 50)
    print("Demo completed! 🎉")


if __name__ == "__main__":
    asyncio.run(main())