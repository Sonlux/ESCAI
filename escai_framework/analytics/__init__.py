"""
Analytics module for the ESCAI framework.

This module provides comprehensive analytics capabilities including:
- Pattern mining algorithms (PrefixSpan, SPADE)
- Machine learning models for prediction (LSTM, Random Forest, XGBoost)
- Statistical analysis and hypothesis testing
- Failure analysis and root cause identification
- Model evaluation and performance assessment
- Online learning with concept drift detection
"""

from .pattern_mining import (
    PrefixSpanMiner,
    SPADEMiner,
    PatternClusterer,
    PatternMiningEngine,
    SequentialPattern,
    PatternMiningConfig
)

from .prediction_models import (
    EnsemblePredictor,
    OnlineLearningPredictor,
    LSTMPredictor,
    HyperparameterTuner,
    ModelConfig
)

from .statistical_analysis import (
    StatisticalAnalyzer,
    StatisticalTest,
    HypothesisTest
)

from .failure_analysis import (
    FailureAnalysisEngine,
    FailurePatternDetector,
    RootCauseAnalyzer,
    FailureMode,
    RootCause,
    FailureAnalysisResult
)

from .model_evaluation import (
    ModelEvaluator,
    ModelPerformance,
    EvaluationResult
)

__all__ = [
    # Pattern Mining
    'PrefixSpanMiner',
    'SPADEMiner',
    'PatternClusterer',
    'PatternMiningEngine',
    'SequentialPattern',
    'PatternMiningConfig',
    
    # Prediction Models
    'EnsemblePredictor',
    'OnlineLearningPredictor',
    'LSTMPredictor',
    'HyperparameterTuner',
    'ModelConfig',
    
    # Statistical Analysis
    'StatisticalAnalyzer',
    'StatisticalTest',
    'HypothesisTest',
    
    # Failure Analysis
    'FailureAnalysisEngine',
    'FailurePatternDetector',
    'RootCauseAnalyzer',
    'FailureMode',
    'RootCause',
    'FailureAnalysisResult',
    
    # Model Evaluation
    'ModelEvaluator',
    'ModelPerformance',
    'EvaluationResult'
]