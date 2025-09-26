"""ESCAI Framework - Epistemic State Cognitive AI Framework."""

__version__ = "0.2.0"
__author__ = "ESCAI Framework Team"

# Lazy imports to avoid circular import issues in CI environments
def __getattr__(name: str):
    """Lazy import for package-level attributes."""
    if name in [
        "EpistemicState",
        "BeliefState", 
        "KnowledgeState",
        "GoalState",
        "BehavioralPattern",
        "ExecutionSequence",
        "CausalRelationship",
        "PredictionResult",
    ]:
        from escai_framework.models import (
            EpistemicState,
            BeliefState,
            KnowledgeState,
            GoalState,
            BehavioralPattern,
            ExecutionSequence,
            CausalRelationship,
            PredictionResult,
        )
        
        # Update module globals
        globals().update({
            "EpistemicState": EpistemicState,
            "BeliefState": BeliefState,
            "KnowledgeState": KnowledgeState,
            "GoalState": GoalState,
            "BehavioralPattern": BehavioralPattern,
            "ExecutionSequence": ExecutionSequence,
            "CausalRelationship": CausalRelationship,
            "PredictionResult": PredictionResult,
        })
        
        return globals()[name]
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "EpistemicState",
    "BeliefState",
    "KnowledgeState",
    "GoalState",
    "BehavioralPattern",
    "ExecutionSequence",
    "CausalRelationship",
    "PredictionResult",
]