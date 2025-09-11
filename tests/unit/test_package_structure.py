"""
Test package structure and imports.
"""

import pytest


class TestPackageStructure:
    """Test that the package structure is correct and imports work."""

    def test_main_package_import(self):
        """Test that the main package can be imported."""
        import escai_framework
        assert hasattr(escai_framework, '__version__')
        assert hasattr(escai_framework, '__author__')

    def test_models_import(self):
        """Test that all models can be imported from the main package."""
        from escai_framework import (
            EpistemicState,
            BeliefState,
            KnowledgeState,
            GoalState,
            BehavioralPattern,
            ExecutionSequence,
            CausalRelationship,
            PredictionResult,
        )
        
        # Verify these are classes
        assert isinstance(EpistemicState, type)
        assert isinstance(BeliefState, type)
        assert isinstance(KnowledgeState, type)
        assert isinstance(GoalState, type)
        assert isinstance(BehavioralPattern, type)
        assert isinstance(ExecutionSequence, type)
        assert isinstance(CausalRelationship, type)
        assert isinstance(PredictionResult, type)

    def test_models_subpackage_import(self):
        """Test that models can be imported from the models subpackage."""
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
        
        # Verify these are classes
        assert isinstance(EpistemicState, type)
        assert isinstance(BeliefState, type)
        assert isinstance(KnowledgeState, type)
        assert isinstance(GoalState, type)
        assert isinstance(BehavioralPattern, type)
        assert isinstance(ExecutionSequence, type)
        assert isinstance(CausalRelationship, type)
        assert isinstance(PredictionResult, type)

    def test_individual_model_imports(self):
        """Test that individual models can be imported directly."""
        from escai_framework.models.epistemic_state import EpistemicState, BeliefState, KnowledgeState, GoalState
        from escai_framework.models.behavioral_pattern import BehavioralPattern, ExecutionSequence
        from escai_framework.models.causal_relationship import CausalRelationship
        from escai_framework.models.prediction_result import PredictionResult
        
        # Verify these are classes
        assert isinstance(EpistemicState, type)
        assert isinstance(BeliefState, type)
        assert isinstance(KnowledgeState, type)
        assert isinstance(GoalState, type)
        assert isinstance(BehavioralPattern, type)
        assert isinstance(ExecutionSequence, type)
        assert isinstance(CausalRelationship, type)
        assert isinstance(PredictionResult, type)

    def test_package_version_consistency(self):
        """Test that package version is consistent."""
        import escai_framework
        from escai_framework import __version__
        
        assert escai_framework.__version__ == __version__
        assert isinstance(__version__, str)
        assert len(__version__) > 0

    def test_all_exports(self):
        """Test that __all__ exports are correct."""
        import escai_framework
        from escai_framework.models import __all__ as models_all
        
        # Check that main package has __all__
        assert hasattr(escai_framework, '__all__')
        
        # Check that models subpackage has __all__
        assert isinstance(models_all, list)
        assert len(models_all) > 0
        
        # Verify all items in __all__ can be imported
        for item in escai_framework.__all__:
            assert hasattr(escai_framework, item)