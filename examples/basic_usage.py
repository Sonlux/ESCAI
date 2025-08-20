#!/usr/bin/env python3
"""
Basic usage example for the ESCAI Framework.

This example demonstrates how to create and use the core data models
for monitoring agent epistemic states, behavioral patterns, causal
relationships, and performance predictions.
"""

from datetime import datetime, timedelta
from escai_framework.models.epistemic_state import (
    EpistemicState, BeliefState, KnowledgeState, GoalState,
    BeliefType, GoalStatus
)
from escai_framework.models.behavioral_pattern import (
    BehavioralPattern, ExecutionSequence, ExecutionStep,
    PatternType, ExecutionStatus
)
from escai_framework.models.causal_relationship import (
    CausalRelationship, CausalEvent, CausalEvidence,
    CausalType, EvidenceType
)
from escai_framework.models.prediction_result import (
    PredictionResult, RiskFactor, Intervention, ConfidenceInterval,
    PredictionType, RiskLevel, InterventionType
)


def demonstrate_epistemic_state():
    """Demonstrate epistemic state monitoring."""
    print("=== Epistemic State Monitoring ===")
    
    # Create beliefs
    belief1 = BeliefState(
        content="The user wants to classify images of cats and dogs",
        belief_type=BeliefType.FACTUAL,
        confidence=0.95,
        evidence=["user input", "task description", "dataset analysis"],
        source="task_analyzer"
    )
    
    belief2 = BeliefState(
        content="The model will achieve >90% accuracy",
        belief_type=BeliefType.PROBABILISTIC,
        confidence=0.75,
        evidence=["historical performance", "model architecture"],
        source="performance_predictor"
    )
    
    # Create knowledge state
    knowledge = KnowledgeState(
        facts=[
            "Dataset contains 10,000 images",
            "Images are 224x224 RGB",
            "Classes are balanced 50/50"
        ],
        rules=[
            "If accuracy < 85%, increase training epochs",
            "If overfitting detected, add dropout",
            "If underfitting detected, increase model complexity"
        ],
        concepts={
            "classification": "supervised learning task",
            "accuracy": "correct predictions / total predictions",
            "overfitting": "high training accuracy, low validation accuracy"
        },
        confidence_score=0.88
    )
    
    # Create goals
    goal1 = GoalState(
        description="Achieve >90% validation accuracy",
        status=GoalStatus.ACTIVE,
        priority=9,
        progress=0.65,
        sub_goals=["preprocess data", "train model", "validate performance"]
    )
    
    goal2 = GoalState(
        description="Complete training within 2 hours",
        status=GoalStatus.ACTIVE,
        priority=6,
        progress=0.40,
        deadline=datetime.utcnow() + timedelta(hours=1.5)
    )
    
    # Create epistemic state
    epistemic_state = EpistemicState(
        agent_id="image_classifier_v1",
        timestamp=datetime.utcnow(),
        belief_states=[belief1, belief2],
        knowledge_state=knowledge,
        goal_states=[goal1, goal2],
        confidence_level=0.82,
        uncertainty_score=0.18,
        decision_context={
            "current_epoch": 15,
            "target_epochs": 50,
            "current_accuracy": 0.87,
            "target_accuracy": 0.90
        }
    )
    
    print(f"Agent: {epistemic_state.agent_id}")
    print(f"Confidence Level: {epistemic_state.confidence_level:.2f}")
    print(f"Uncertainty Score: {epistemic_state.uncertainty_score:.2f}")
    print(f"Active Goals: {len([g for g in epistemic_state.goal_states if g.status == GoalStatus.ACTIVE])}")
    print(f"Beliefs: {len(epistemic_state.belief_states)}")
    print(f"Knowledge Facts: {len(epistemic_state.knowledge_state.facts)}")
    
    # Serialize to JSON
    json_data = epistemic_state.to_json()
    print(f"JSON serialization successful: {len(json_data)} characters")
    print()


def demonstrate_behavioral_pattern():
    """Demonstrate behavioral pattern analysis."""
    print("=== Behavioral Pattern Analysis ===")
    
    # Create execution steps
    steps = [
        ExecutionStep(
            step_id="step_001",
            action="load_dataset",
            timestamp=datetime.utcnow(),
            duration_ms=2500,
            status=ExecutionStatus.SUCCESS,
            inputs={"dataset_path": "/data/cats_dogs.zip"},
            outputs={"samples_loaded": 10000, "classes": ["cat", "dog"]}
        ),
        ExecutionStep(
            step_id="step_002",
            action="preprocess_images",
            timestamp=datetime.utcnow() + timedelta(seconds=3),
            duration_ms=15000,
            status=ExecutionStatus.SUCCESS,
            inputs={"resize_to": [224, 224], "normalize": True},
            outputs={"processed_samples": 10000}
        ),
        ExecutionStep(
            step_id="step_003",
            action="initialize_model",
            timestamp=datetime.utcnow() + timedelta(seconds=18),
            duration_ms=1200,
            status=ExecutionStatus.SUCCESS,
            inputs={"architecture": "ResNet50", "pretrained": True},
            outputs={"model_parameters": 25636712}
        ),
        ExecutionStep(
            step_id="step_004",
            action="train_model",
            timestamp=datetime.utcnow() + timedelta(seconds=20),
            duration_ms=120000,
            status=ExecutionStatus.SUCCESS,
            inputs={"epochs": 10, "batch_size": 32, "learning_rate": 0.001},
            outputs={"final_accuracy": 0.89, "final_loss": 0.23}
        )
    ]
    
    # Create execution sequence
    sequence = ExecutionSequence(
        sequence_id="training_seq_001",
        agent_id="image_classifier_v1",
        task_description="Train image classification model on cats vs dogs dataset",
        steps=steps
    )
    sequence.calculate_metrics()
    
    # Create behavioral pattern
    pattern = BehavioralPattern(
        pattern_id="standard_training_pattern",
        pattern_name="Standard Image Classification Training",
        pattern_type=PatternType.SEQUENTIAL,
        description="Standard workflow for training image classification models",
        execution_sequences=[sequence],
        common_triggers=["new_dataset", "model_update_request", "accuracy_below_threshold"],
        failure_modes=["out_of_memory", "data_corruption", "convergence_failure"]
    )
    pattern.calculate_statistics()
    
    print(f"Pattern: {pattern.pattern_name}")
    print(f"Type: {pattern.pattern_type.value}")
    print(f"Frequency: {pattern.frequency}")
    print(f"Success Rate: {pattern.success_rate:.2f}")
    print(f"Average Duration: {pattern.average_duration_ms/1000:.1f} seconds")
    print(f"Common Triggers: {', '.join(pattern.common_triggers)}")
    print()


def demonstrate_causal_relationship():
    """Demonstrate causal relationship discovery."""
    print("=== Causal Relationship Analysis ===")
    
    # Create cause event
    cause_event = CausalEvent(
        event_id="cause_lr_reduction",
        event_type="parameter_adjustment",
        description="Learning rate reduced from 0.01 to 0.001",
        timestamp=datetime.utcnow(),
        agent_id="image_classifier_v1",
        context={"epoch": 25, "validation_loss": 0.45, "training_loss": 0.12},
        attributes={"old_lr": 0.01, "new_lr": 0.001, "reduction_factor": 0.1}
    )
    
    # Create effect event
    effect_event = CausalEvent(
        event_id="effect_accuracy_improvement",
        event_type="performance_change",
        description="Validation accuracy improved from 0.82 to 0.89",
        timestamp=datetime.utcnow() + timedelta(minutes=5),
        agent_id="image_classifier_v1",
        context={"epoch": 30, "validation_loss": 0.28, "training_loss": 0.15},
        attributes={"old_accuracy": 0.82, "new_accuracy": 0.89, "improvement": 0.07}
    )
    
    # Create evidence
    evidence = CausalEvidence(
        evidence_type=EvidenceType.STATISTICAL,
        description="Strong correlation between learning rate reduction and accuracy improvement",
        strength=0.87,
        confidence=0.92,
        source="correlation_analysis",
        statistical_measures={
            "correlation_coefficient": 0.87,
            "p_value": 0.003,
            "confidence_interval": [0.75, 0.94]
        }
    )
    
    # Create causal relationship
    relationship = CausalRelationship(
        relationship_id="lr_reduction_accuracy_improvement",
        cause_event=cause_event,
        effect_event=effect_event,
        causal_type=CausalType.DIRECT,
        strength=0.85,
        confidence=0.90,
        delay_ms=300000,  # 5 minutes
        evidence=[evidence],
        statistical_significance=0.003,
        effect_size=0.07,
        causal_mechanism="Lower learning rate allows more precise weight updates, reducing overfitting",
        confounding_factors=["batch_size", "model_architecture", "data_augmentation"]
    )
    
    print(f"Causal Relationship: {relationship.relationship_id}")
    print(f"Type: {relationship.causal_type.value}")
    print(f"Strength: {relationship.strength:.2f}")
    print(f"Confidence: {relationship.confidence:.2f}")
    print(f"Delay: {relationship.delay_ms/1000:.0f} seconds")
    print(f"Effect Size: {relationship.effect_size:.3f}")
    print(f"Mechanism: {relationship.causal_mechanism}")
    print()


def demonstrate_prediction_result():
    """Demonstrate performance prediction."""
    print("=== Performance Prediction ===")
    
    # Create risk factors
    risk1 = RiskFactor(
        factor_id="high_complexity_risk",
        name="High Model Complexity",
        description="Model has 25M+ parameters, increasing overfitting risk",
        impact_score=0.7,
        probability=0.6,
        category="model_architecture",
        mitigation_strategies=["add_dropout", "reduce_model_size", "increase_regularization"]
    )
    
    risk2 = RiskFactor(
        factor_id="limited_data_risk",
        name="Limited Training Data",
        description="Only 10K samples may not be sufficient for complex model",
        impact_score=0.6,
        probability=0.4,
        category="data_quality",
        mitigation_strategies=["data_augmentation", "transfer_learning", "collect_more_data"]
    )
    
    # Create interventions
    intervention1 = Intervention(
        intervention_id="add_dropout_intervention",
        intervention_type=InterventionType.PARAMETER_ADJUSTMENT,
        name="Add Dropout Layers",
        description="Add dropout layers with 0.5 probability to reduce overfitting",
        expected_impact=0.8,
        implementation_cost=0.2,
        urgency=RiskLevel.MEDIUM,
        parameters={"dropout_rate": 0.5, "layers": ["fc1", "fc2"]},
        prerequisites=["model_architecture_access", "training_pipeline_control"]
    )
    
    intervention2 = Intervention(
        intervention_id="data_augmentation_intervention",
        intervention_type=InterventionType.STRATEGY_CHANGE,
        name="Implement Data Augmentation",
        description="Add rotation, flip, and color jittering to increase effective dataset size",
        expected_impact=0.6,
        implementation_cost=0.3,
        urgency=RiskLevel.LOW,
        parameters={
            "rotation_range": 15,
            "horizontal_flip": True,
            "color_jitter": 0.2
        }
    )
    
    # Create confidence interval
    confidence_interval = ConfidenceInterval(
        lower_bound=0.85,
        upper_bound=0.93,
        confidence_level=0.95
    )
    
    # Create prediction result
    prediction = PredictionResult(
        prediction_id="accuracy_prediction_001",
        agent_id="image_classifier_v1",
        prediction_type=PredictionType.SUCCESS_PROBABILITY,
        predicted_value=0.89,
        confidence_score=0.87,
        confidence_interval=confidence_interval,
        risk_factors=[risk1, risk2],
        recommended_interventions=[intervention1, intervention2],
        model_name="RandomForestPredictor",
        model_version="1.2.0",
        features_used=[
            "model_complexity", "dataset_size", "training_epochs",
            "learning_rate", "batch_size", "validation_split"
        ],
        prediction_horizon_ms=3600000,  # 1 hour
        created_at=datetime.utcnow()
    )
    
    print(f"Prediction: {prediction.prediction_type.value}")
    print(f"Predicted Value: {prediction.predicted_value:.3f}")
    print(f"Confidence: {prediction.confidence_score:.2f}")
    print(f"Confidence Interval: [{confidence_interval.lower_bound:.2f}, {confidence_interval.upper_bound:.2f}]")
    print(f"Risk Level: {prediction.risk_level.value}")
    print(f"Overall Risk Score: {prediction.calculate_overall_risk_score():.3f}")
    print(f"Risk Factors: {len(prediction.risk_factors)}")
    print(f"Recommended Interventions: {len(prediction.recommended_interventions)}")
    
    # Show top intervention by benefit/cost ratio
    if prediction.recommended_interventions:
        top_intervention = max(
            prediction.recommended_interventions,
            key=lambda x: x.calculate_benefit_cost_ratio()
        )
        print(f"Top Intervention: {top_intervention.name} (B/C ratio: {top_intervention.calculate_benefit_cost_ratio():.1f})")
    print()


def main():
    """Run all demonstrations."""
    print("ESCAI Framework - Basic Usage Examples")
    print("=" * 50)
    print()
    
    demonstrate_epistemic_state()
    demonstrate_behavioral_pattern()
    demonstrate_causal_relationship()
    demonstrate_prediction_result()
    
    print("=" * 50)
    print("All examples completed successfully!")
    print("Check the ESCAI Framework documentation for more advanced usage patterns.")


if __name__ == "__main__":
    main()