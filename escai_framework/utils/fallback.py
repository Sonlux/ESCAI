"""
Fallback mechanisms for NLP and ML model failures with graceful degradation.

This module provides fallback strategies when primary processing methods fail,
ensuring the system continues to operate with reduced functionality rather than
complete failure.
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum

from ..models.epistemic_state import EpistemicState, BeliefState, KnowledgeState, GoalState
from ..models.behavioral_pattern import BehavioralPattern, ExecutionSequence
from ..models.causal_relationship import CausalRelationship
from ..models.prediction_result import PredictionResult
from .exceptions import ProcessingError, ModelLoadError, EpistemicExtractionError


logger = logging.getLogger(__name__)


class FallbackStrategy(Enum):
    """Available fallback strategies."""
    RULE_BASED = "rule_based"
    SIMPLIFIED_MODEL = "simplified_model"
    CACHED_RESULT = "cached_result"
    DEFAULT_VALUE = "default_value"
    STATISTICAL_BASELINE = "statistical_baseline"


@dataclass
class FallbackResult:
    """Result from a fallback operation."""
    success: bool
    result: Any
    strategy_used: FallbackStrategy
    confidence: float
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if not self.success and not self.error_message:
            self.error_message = "Fallback operation failed"


class FallbackProvider(ABC):
    """Abstract base class for fallback providers."""
    
    @abstractmethod
    def can_handle(self, input_data: Any, error: Exception) -> bool:
        """Check if this provider can handle the given input and error."""
        pass
    
    @abstractmethod
    async def execute_fallback(self, input_data: Any, error: Exception) -> FallbackResult:
        """Execute the fallback strategy."""
        pass
    
    @property
    @abstractmethod
    def strategy(self) -> FallbackStrategy:
        """Return the fallback strategy this provider implements."""
        pass


class RuleBasedEpistemicExtractor(FallbackProvider):
    """Rule-based fallback for epistemic state extraction."""
    
    def __init__(self):
        # Confidence patterns
        self.confidence_patterns = [
            (r'\b(?:very\s+)?confident\b', 0.9),
            (r'\b(?:quite\s+)?sure\b', 0.8),
            (r'\blikely\b', 0.7),
            (r'\bprobably\b', 0.6),
            (r'\bmaybe\b', 0.4),
            (r'\bunsure\b', 0.3),
            (r'\bdoubtful\b', 0.2),
            (r'\bunlikely\b', 0.1)
        ]
        
        # Belief indicators
        self.belief_patterns = [
            r'\bi\s+(?:believe|think|assume|suppose)\s+(?:that\s+)?(.+)',
            r'\bit\s+(?:seems|appears)\s+(?:that\s+)?(.+)',
            r'\bmy\s+understanding\s+is\s+(?:that\s+)?(.+)',
            r'\bi\s+(?:am\s+)?convinced\s+(?:that\s+)?(.+)'
        ]
        
        # Goal indicators
        self.goal_patterns = [
            r'\bi\s+(?:want|need|aim)\s+to\s+(.+)',
            r'\bmy\s+(?:goal|objective|target)\s+is\s+(?:to\s+)?(.+)',
            r'\bi\s+(?:plan|intend)\s+to\s+(.+)',
            r'\bthe\s+(?:goal|objective)\s+is\s+(?:to\s+)?(.+)'
        ]
    
    def can_handle(self, input_data: Any, error: Exception) -> bool:
        """Check if we can handle epistemic extraction fallback."""
        return isinstance(error, (EpistemicExtractionError, ModelLoadError)) and isinstance(input_data, (str, list))
    
    async def execute_fallback(self, input_data: Any, error: Exception) -> FallbackResult:
        """Execute rule-based epistemic extraction."""
        try:
            if isinstance(input_data, list):
                text = " ".join(str(item) for item in input_data)
            else:
                text = str(input_data)
            
            # Extract beliefs using patterns
            beliefs: List[str] = []
            for pattern in self.belief_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    belief_text = match.group(1).strip()
                    confidence = self._extract_confidence(belief_text)
                    beliefs.append(BeliefState(
                        content=belief_text,
                        confidence=confidence,
                        source="rule_based_extraction"
                    ))
            
            # Extract goals using patterns
            goals: List[str] = []
            for pattern in self.goal_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    goal_text = match.group(1).strip()
                    goals.append(goal_text)
            
            # Create simplified epistemic state
            epistemic_state = EpistemicState(
                agent_id="unknown",
                belief_states=beliefs,
                knowledge_state=KnowledgeState(
                    facts=[],
                    confidence=0.5,
                    source="rule_based_fallback"
                ),
                goal_state=GoalState(
                    primary_goals=goals[:3],  # Take first 3 goals
                    secondary_goals=goals[3:],
                    completion_status={}
                ),
                confidence_level=0.6,  # Lower confidence for rule-based extraction
                uncertainty_score=0.4
            )
            
            return FallbackResult(
                success=True,
                result=epistemic_state,
                strategy_used=FallbackStrategy.RULE_BASED,
                confidence=0.6
            )
            
        except Exception as e:
            logger.error(f"Rule-based epistemic extraction failed: {e}")
            return FallbackResult(
                success=False,
                result=None,
                strategy_used=FallbackStrategy.RULE_BASED,
                confidence=0.0,
                error_message=str(e)
            )
    
    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score from text using patterns."""
        text_lower = text.lower()
        for pattern, confidence in self.confidence_patterns:
            if re.search(pattern, text_lower):
                return confidence
        return 0.5  # Default confidence
    
    @property
    def strategy(self) -> FallbackStrategy:
        return FallbackStrategy.RULE_BASED


class StatisticalPatternAnalyzer(FallbackProvider):
    """Statistical fallback for behavioral pattern analysis."""
    
    def can_handle(self, input_data: Any, error: Exception) -> bool:
        """Check if we can handle pattern analysis fallback."""
        return isinstance(error, ProcessingError) and isinstance(input_data, list)
    
    async def execute_fallback(self, input_data: Any, error: Exception) -> FallbackResult:
        """Execute statistical pattern analysis."""
        try:
            sequences = input_data if isinstance(input_data, list) else [input_data]
            
            # Simple frequency-based pattern detection
            action_counts: Dict[str, int] = {}
            sequence_lengths: List[int] = []
            
            for seq in sequences:
                if hasattr(seq, 'actions'):
                    actions = seq.actions
                elif isinstance(seq, dict) and 'actions' in seq:
                    actions = seq['actions']
                else:
                    actions = [str(seq)]
                
                sequence_lengths.append(len(actions))
                
                for action in actions:
                    action_str = str(action)
                    action_counts[action_str] = action_counts.get(action_str, 0) + 1
            
            # Create simple patterns based on frequency
            patterns: List[Dict[str, Any]] = []
            total_actions = sum(action_counts.values())
            
            for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
                if count >= 2:  # Only include actions that appear multiple times
                    frequency = count / len(sequences) if sequences else 0
                    patterns.append(BehavioralPattern(
                        pattern_id=f"freq_{action}_{count}",
                        pattern_name=f"Frequent action: {action}",
                        frequency=frequency,
                        success_rate=0.5,  # Default success rate
                        average_duration=sum(sequence_lengths) / len(sequence_lengths) if sequence_lengths else 0,
                        common_triggers=[action],
                        failure_modes=[],
                        statistical_significance=min(frequency, 0.8)  # Cap at 0.8 for statistical fallback
                    ))
            
            return FallbackResult(
                success=True,
                result=patterns,
                strategy_used=FallbackStrategy.STATISTICAL_BASELINE,
                confidence=0.5
            )
            
        except Exception as e:
            logger.error(f"Statistical pattern analysis failed: {e}")
            return FallbackResult(
                success=False,
                result=[],
                strategy_used=FallbackStrategy.STATISTICAL_BASELINE,
                confidence=0.0,
                error_message=str(e)
            )
    
    @property
    def strategy(self) -> FallbackStrategy:
        return FallbackStrategy.STATISTICAL_BASELINE


class SimpleCausalAnalyzer(FallbackProvider):
    """Simple temporal-based causal analysis fallback."""
    
    def can_handle(self, input_data: Any, error: Exception) -> bool:
        """Check if we can handle causal analysis fallback."""
        return isinstance(error, ProcessingError) and isinstance(input_data, list)
    
    async def execute_fallback(self, input_data: Any, error: Exception) -> FallbackResult:
        """Execute simple temporal causal analysis."""
        try:
            events = input_data if isinstance(input_data, list) else [input_data]
            
            # Simple temporal causality: if A happens before B consistently, assume A causes B
            causal_relationships: List[Dict[str, Any]] = []
            event_pairs: Dict[str, List[str]] = {}
            
            for i, event in enumerate(events[:-1]):
                next_event = events[i + 1]
                
                event_str = str(event)
                next_event_str = str(next_event)
                
                pair_key = (event_str, next_event_str)
                if pair_key not in event_pairs:
                    event_pairs[pair_key] = 0
                event_pairs[pair_key] += 1
            
            # Create causal relationships for frequent pairs
            total_pairs = len(events) - 1 if len(events) > 1 else 1
            
            for (cause, effect), count in event_pairs.items():
                if count >= 2:  # Must occur at least twice
                    strength = count / total_pairs
                    causal_relationships.append(CausalRelationship(
                        cause_event=cause,
                        effect_event=effect,
                        strength=strength,
                        confidence=min(strength * 0.8, 0.6),  # Conservative confidence
                        delay_ms=1000,  # Default 1 second delay
                        evidence=[f"Temporal sequence observed {count} times"],
                        statistical_significance=min(strength, 0.5)  # Cap significance for simple analysis
                    ))
            
            return FallbackResult(
                success=True,
                result=causal_relationships,
                strategy_used=FallbackStrategy.RULE_BASED,
                confidence=0.4
            )
            
        except Exception as e:
            logger.error(f"Simple causal analysis failed: {e}")
            return FallbackResult(
                success=False,
                result=[],
                strategy_used=FallbackStrategy.RULE_BASED,
                confidence=0.0,
                error_message=str(e)
            )
    
    @property
    def strategy(self) -> FallbackStrategy:
        return FallbackStrategy.RULE_BASED


class BaselinePredictor(FallbackProvider):
    """Baseline prediction fallback using simple heuristics."""
    
    def can_handle(self, input_data: Any, error: Exception) -> bool:
        """Check if we can handle prediction fallback."""
        return isinstance(error, (ProcessingError, ModelLoadError))
    
    async def execute_fallback(self, input_data: Any, error: Exception) -> FallbackResult:
        """Execute baseline prediction."""
        try:
            # Simple heuristic: assume 50% success rate with high uncertainty
            prediction = PredictionResult(
                agent_id="unknown",
                prediction_type="success_probability",
                predicted_value=0.5,
                confidence=0.3,
                uncertainty=0.7,
                risk_factors=["Insufficient data for accurate prediction"],
                recommended_actions=["Collect more data", "Monitor closely"],
                model_used="baseline_fallback"
            )
            
            return FallbackResult(
                success=True,
                result=prediction,
                strategy_used=FallbackStrategy.DEFAULT_VALUE,
                confidence=0.3
            )
            
        except Exception as e:
            logger.error(f"Baseline prediction failed: {e}")
            return FallbackResult(
                success=False,
                result=None,
                strategy_used=FallbackStrategy.DEFAULT_VALUE,
                confidence=0.0,
                error_message=str(e)
            )
    
    @property
    def strategy(self) -> FallbackStrategy:
        return FallbackStrategy.DEFAULT_VALUE


class FallbackManager:
    """Manages fallback strategies for different types of processing failures."""
    
    def __init__(self):
        self.providers: List[FallbackProvider] = [
            RuleBasedEpistemicExtractor(),
            StatisticalPatternAnalyzer(),
            SimpleCausalAnalyzer(),
            BaselinePredictor()
        ]
        self._cache: Dict[str, FallbackResult] = {}
    
    def register_provider(self, provider: FallbackProvider):
        """Register a new fallback provider."""
        self.providers.append(provider)
        logger.info(f"Registered fallback provider: {provider.__class__.__name__}")
    
    async def execute_with_fallback(
        self,
        primary_func: Callable,
        input_data: Any,
        cache_key: Optional[str] = None,
        *args,
        **kwargs
    ) -> FallbackResult:
        """
        Execute a function with fallback support.
        
        Args:
            primary_func: The primary function to execute
            input_data: Input data for the function
            cache_key: Optional cache key for storing results
            *args, **kwargs: Additional arguments for the primary function
        
        Returns:
            FallbackResult containing the result and metadata
        """
        # Check cache first
        if cache_key and cache_key in self._cache:
            cached_result = self._cache[cache_key]
            logger.info(f"Using cached fallback result for key: {cache_key}")
            return cached_result
        
        # Try primary function first
        try:
            result = await primary_func(input_data, *args, **kwargs)
            return FallbackResult(
                success=True,
                result=result,
                strategy_used=FallbackStrategy.RULE_BASED,  # Assuming primary is rule-based
                confidence=1.0
            )
        except Exception as primary_error:
            logger.warning(f"Primary function failed: {primary_error}. Trying fallbacks...")
            
            # Try fallback providers
            for provider in self.providers:
                if provider.can_handle(input_data, primary_error):
                    try:
                        fallback_result = await provider.execute_fallback(input_data, primary_error)
                        
                        if fallback_result.success:
                            logger.info(
                                f"Fallback successful using {provider.strategy.value} "
                                f"with confidence {fallback_result.confidence:.2f}"
                            )
                            
                            # Cache successful fallback result
                            if cache_key:
                                self._cache[cache_key] = fallback_result
                            
                            return fallback_result
                        else:
                            logger.warning(f"Fallback provider {provider.__class__.__name__} failed")
                            
                    except Exception as fallback_error:
                        logger.error(f"Fallback provider {provider.__class__.__name__} raised exception: {fallback_error}")
                        continue
            
            # All fallbacks failed
            logger.error("All fallback strategies failed")
            return FallbackResult(
                success=False,
                result=None,
                strategy_used=FallbackStrategy.DEFAULT_VALUE,
                confidence=0.0,
                error_message=f"Primary function failed: {primary_error}. All fallbacks failed."
            )
    
    def clear_cache(self):
        """Clear the fallback result cache."""
        self._cache.clear()
        logger.info("Fallback cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "cached_keys": list(self._cache.keys())
        }


# Global fallback manager instance
_fallback_manager = FallbackManager()


def get_fallback_manager() -> FallbackManager:
    """Get the global fallback manager instance."""
    return _fallback_manager


def register_fallback_provider(provider: FallbackProvider):
    """Register a fallback provider with the global manager."""
    _fallback_manager.register_provider(provider)


async def execute_with_fallback(
    primary_func: Callable,
    input_data: Any,
    cache_key: Optional[str] = None,
    *args,
    **kwargs
) -> FallbackResult:
    """Execute a function with fallback support using the global manager."""
    return await _fallback_manager.execute_with_fallback(
        primary_func, input_data, cache_key, *args, **kwargs
    )