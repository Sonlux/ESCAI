"""
Causal inference engine for the ESCAI framework.

This module implements causal analysis capabilities including temporal causality detection,
Granger causality testing, structural causal model construction, counterfactual reasoning,
and intervention effect estimation.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Statistical analysis
try:
    from statsmodels.tsa.stattools import grangercausalitytests
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. Granger causality testing will be limited.")

# Causal inference
try:
    import dowhy
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False
    warnings.warn("DoWhy not available. Structural causal models will be limited.")

from ..models.causal_relationship import (
    CausalRelationship, CausalEvent, CausalEvidence, CausalType, EvidenceType
)
from ..utils.validation import validate_time_series_data


logger = logging.getLogger(__name__)


class TemporalEvent:
    """Represents a temporal event for causal analysis."""
    
    def __init__(self, event_id: str, event_type: str, timestamp: datetime, 
                 agent_id: str, attributes: Dict[str, Any] = None):
        self.event_id = event_id
        self.event_type = event_type
        self.timestamp = timestamp
        self.agent_id = agent_id
        self.attributes = attributes or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "attributes": self.attributes
        }


class GrangerResult:
    """Results from Granger causality testing."""
    
    def __init__(self, cause_variable: str, effect_variable: str, 
                 p_values: Dict[int, float], f_statistics: Dict[int, float],
                 optimal_lag: int, is_causal: bool, confidence: float):
        self.cause_variable = cause_variable
        self.effect_variable = effect_variable
        self.p_values = p_values
        self.f_statistics = f_statistics
        self.optimal_lag = optimal_lag
        self.is_causal = is_causal
        self.confidence = confidence
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cause_variable": self.cause_variable,
            "effect_variable": self.effect_variable,
            "p_values": self.p_values,
            "f_statistics": self.f_statistics,
            "optimal_lag": self.optimal_lag,
            "is_causal": self.is_causal,
            "confidence": self.confidence
        }


class CausalGraph:
    """Represents a causal graph structure."""
    
    def __init__(self):
        self.nodes = set()
        self.edges = []  # List of (cause, effect, strength) tuples
        self.relationships = {}  # Dict mapping edge to CausalRelationship
    
    def add_relationship(self, relationship: CausalRelationship):
        """Add a causal relationship to the graph."""
        cause_id = relationship.cause_event.event_id
        effect_id = relationship.effect_event.event_id
        
        self.nodes.add(cause_id)
        self.nodes.add(effect_id)
        
        edge = (cause_id, effect_id)
        self.edges.append((cause_id, effect_id, relationship.strength))
        self.relationships[edge] = relationship
    
    def get_ancestors(self, node: str) -> List[str]:
        """Get all ancestor nodes (causes) of a given node."""
        ancestors = []
        for cause, effect, _ in self.edges:
            if effect == node:
                ancestors.append(cause)
                ancestors.extend(self.get_ancestors(cause))
        return list(set(ancestors))
    
    def get_descendants(self, node: str) -> List[str]:
        """Get all descendant nodes (effects) of a given node."""
        descendants = []
        for cause, effect, _ in self.edges:
            if cause == node:
                descendants.append(effect)
                descendants.extend(self.get_descendants(effect))
        return list(set(descendants))


class InterventionEffect:
    """Results from intervention analysis."""
    
    def __init__(self, intervention_variable: str, target_variable: str,
                 intervention_value: Any, expected_effect: float,
                 confidence_interval: Tuple[float, float], p_value: float):
        self.intervention_variable = intervention_variable
        self.target_variable = target_variable
        self.intervention_value = intervention_value
        self.expected_effect = expected_effect
        self.confidence_interval = confidence_interval
        self.p_value = p_value
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "intervention_variable": self.intervention_variable,
            "target_variable": self.target_variable,
            "intervention_value": self.intervention_value,
            "expected_effect": self.expected_effect,
            "confidence_interval": self.confidence_interval,
            "p_value": self.p_value
        }


class CausalEngine:
    """
    Main causal inference engine for analyzing causal relationships in agent behavior.
    
    This engine provides capabilities for:
    - Temporal causality detection
    - Granger causality testing
    - Structural causal model construction
    - Counterfactual reasoning
    - Intervention effect estimation
    """
    
    def __init__(self, significance_threshold: float = 0.05, 
                 max_lag: int = 10, min_observations: int = 50):
        """
        Initialize the causal engine.
        
        Args:
            significance_threshold: P-value threshold for statistical significance
            max_lag: Maximum lag to test for Granger causality
            min_observations: Minimum number of observations required for analysis
        """
        self.significance_threshold = significance_threshold
        self.max_lag = max_lag
        self.min_observations = min_observations
        self.logger = logging.getLogger(__name__)
        
        # Check for required dependencies
        if not STATSMODELS_AVAILABLE:
            self.logger.warning("statsmodels not available. Some functionality will be limited.")
        if not DOWHY_AVAILABLE:
            self.logger.warning("DoWhy not available. Structural causal models will be limited.")
    
    async def discover_relationships_from_events(self, events: List[TemporalEvent]) -> List[CausalRelationship]:
        """
        Discover causal relationships from a sequence of temporal events.
        
        Args:
            events: List of temporal events to analyze
            
        Returns:
            List of discovered causal relationships
        """
        if len(events) < self.min_observations:
            self.logger.warning(f"Insufficient events for analysis: {len(events)} < {self.min_observations}")
            return []
        
        relationships = []
        
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        # Group events by type and agent
        event_groups = self._group_events(sorted_events)
        
        # Detect temporal relationships
        temporal_relationships = await self._detect_temporal_relationships(event_groups)
        relationships.extend(temporal_relationships)
        
        # Perform statistical analysis if we have enough data
        if len(sorted_events) >= self.min_observations:
            statistical_relationships = await self._detect_statistical_relationships(event_groups)
            relationships.extend(statistical_relationships)
        
        return relationships
    
    async def test_granger_causality(self, time_series: pd.DataFrame, 
                                   cause_column: str, effect_column: str) -> GrangerResult:
        """
        Test for Granger causality between two time series.
        
        Args:
            time_series: DataFrame with time series data
            cause_column: Name of the potential cause variable
            effect_column: Name of the potential effect variable
            
        Returns:
            GrangerResult with test results
        """
        if not STATSMODELS_AVAILABLE:
            raise RuntimeError("statsmodels is required for Granger causality testing")
        
        # Validate input data
        if cause_column not in time_series.columns or effect_column not in time_series.columns:
            raise ValueError(f"Columns {cause_column} or {effect_column} not found in time series")
        
        if len(time_series) < self.min_observations:
            raise ValueError(f"Insufficient observations: {len(time_series)} < {self.min_observations}")
        
        # Prepare data for Granger test
        data = time_series[[effect_column, cause_column]].dropna()
        
        if len(data) < self.min_observations:
            raise ValueError("Insufficient valid observations after removing NaN values")
        
        try:
            # Perform Granger causality test
            max_lag = min(self.max_lag, len(data) // 4)  # Ensure we have enough observations
            test_result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
            
            # Extract results
            p_values = {}
            f_statistics = {}
            
            for lag in range(1, max_lag + 1):
                if lag in test_result:
                    p_val = test_result[lag][0]['ssr_ftest'][1]  # p-value
                    f_stat = test_result[lag][0]['ssr_ftest'][0]  # F-statistic
                    p_values[lag] = p_val
                    f_statistics[lag] = f_stat
            
            # Find optimal lag (lowest p-value)
            if p_values:
                optimal_lag = min(p_values.keys(), key=lambda k: p_values[k])
                min_p_value = p_values[optimal_lag]
                is_causal = min_p_value < self.significance_threshold
                confidence = 1.0 - min_p_value
            else:
                optimal_lag = 1
                is_causal = False
                confidence = 0.0
            
            return GrangerResult(
                cause_variable=cause_column,
                effect_variable=effect_column,
                p_values=p_values,
                f_statistics=f_statistics,
                optimal_lag=optimal_lag,
                is_causal=is_causal,
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Error in Granger causality test: {e}")
            return GrangerResult(
                cause_variable=cause_column,
                effect_variable=effect_column,
                p_values={},
                f_statistics={},
                optimal_lag=1,
                is_causal=False,
                confidence=0.0
            )
    
    async def build_causal_graph(self, relationships: List[CausalRelationship]) -> CausalGraph:
        """
        Build a causal graph from discovered relationships.
        
        Args:
            relationships: List of causal relationships
            
        Returns:
            CausalGraph representing the relationships
        """
        graph = CausalGraph()
        
        for relationship in relationships:
            if relationship.confidence > 0.5:  # Only include confident relationships
                graph.add_relationship(relationship)
        
        return graph
    
    async def analyze_interventions(self, graph: CausalGraph, 
                                  intervention_variable: str, 
                                  intervention_value: Any,
                                  target_variable: str,
                                  data: pd.DataFrame) -> InterventionEffect:
        """
        Analyze the effect of an intervention on a target variable.
        
        Args:
            graph: Causal graph
            intervention_variable: Variable to intervene on
            intervention_value: Value to set for intervention
            target_variable: Variable to measure effect on
            data: Historical data for analysis
            
        Returns:
            InterventionEffect with estimated effects
        """
        try:
            # Simple intervention analysis using regression
            # In a full implementation, this would use more sophisticated causal inference
            
            # Find path from intervention to target
            if intervention_variable not in graph.nodes or target_variable not in graph.nodes:
                return InterventionEffect(
                    intervention_variable=intervention_variable,
                    target_variable=target_variable,
                    intervention_value=intervention_value,
                    expected_effect=0.0,
                    confidence_interval=(0.0, 0.0),
                    p_value=1.0
                )
            
            # Use regression to estimate effect
            if intervention_variable in data.columns and target_variable in data.columns:
                X = data[[intervention_variable]].values
                y = data[target_variable].values
                
                # Remove NaN values
                mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
                X_clean = X[mask]
                y_clean = y[mask]
                
                if len(X_clean) < 10:  # Need minimum observations
                    return InterventionEffect(
                        intervention_variable=intervention_variable,
                        target_variable=target_variable,
                        intervention_value=intervention_value,
                        expected_effect=0.0,
                        confidence_interval=(0.0, 0.0),
                        p_value=1.0
                    )
                
                # Simple linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    X_clean.flatten(), y_clean
                )
                
                # Calculate expected effect
                current_mean = np.mean(X_clean)
                expected_effect = slope * (intervention_value - current_mean)
                
                # Calculate confidence interval (rough approximation)
                margin_error = 1.96 * std_err * abs(intervention_value - current_mean)
                confidence_interval = (
                    expected_effect - margin_error,
                    expected_effect + margin_error
                )
                
                return InterventionEffect(
                    intervention_variable=intervention_variable,
                    target_variable=target_variable,
                    intervention_value=intervention_value,
                    expected_effect=expected_effect,
                    confidence_interval=confidence_interval,
                    p_value=p_value
                )
            
        except Exception as e:
            self.logger.error(f"Error in intervention analysis: {e}")
        
        return InterventionEffect(
            intervention_variable=intervention_variable,
            target_variable=target_variable,
            intervention_value=intervention_value,
            expected_effect=0.0,
            confidence_interval=(0.0, 0.0),
            p_value=1.0
        )
    
    def _group_events(self, events: List[TemporalEvent]) -> Dict[str, List[TemporalEvent]]:
        """Group events by type for analysis."""
        groups: Dict[str, List[TemporalEvent]] = {}
        for event in events:
            event_type = event.event_type
            if event_type not in groups:
                groups[event_type] = []
            groups[event_type].append(event)
        return groups
    
    async def _detect_temporal_relationships(self, 
                                           event_groups: Dict[str, List[TemporalEvent]]) -> List[CausalRelationship]:
        """Detect temporal causal relationships between event types."""
        relationships = []
        event_types = list(event_groups.keys())
        
        # Look for temporal patterns between different event types
        for i, cause_type in enumerate(event_types):
            for j, effect_type in enumerate(event_types):
                if i != j:  # Don't compare event type with itself
                    relationship = await self._analyze_temporal_pattern(
                        event_groups[cause_type], event_groups[effect_type],
                        cause_type, effect_type
                    )
                    if relationship:
                        relationships.append(relationship)
        
        return relationships
    
    async def _analyze_temporal_pattern(self, cause_events: List[TemporalEvent],
                                      effect_events: List[TemporalEvent],
                                      cause_type: str, effect_type: str) -> Optional[CausalRelationship]:
        """Analyze temporal patterns between two event types."""
        if not cause_events or not effect_events:
            return None
        
        # Find temporal correlations
        temporal_correlations = []
        
        for cause_event in cause_events:
            for effect_event in effect_events:
                if effect_event.timestamp > cause_event.timestamp:
                    delay = (effect_event.timestamp - cause_event.timestamp).total_seconds() * 1000
                    if delay < 300000:  # Within 5 minutes
                        temporal_correlations.append({
                            'cause': cause_event,
                            'effect': effect_event,
                            'delay': delay
                        })
        
        if len(temporal_correlations) < 3:  # Need minimum correlations
            return None
        
        # Calculate statistics
        delays = [float(corr['delay']) if isinstance(corr['delay'], (int, float, str)) else 0.0 for corr in temporal_correlations]
        avg_delay = float(np.mean(np.array(delays, dtype=float)))
        delay_std = float(np.std(np.array(delays, dtype=float)))
        
        # Calculate strength based on frequency and consistency
        total_cause_events = len(cause_events)
        correlation_count = len(temporal_correlations)
        frequency_strength = min(1.0, correlation_count / total_cause_events)
        consistency_strength = float(max(0.0, 1.0 - (delay_std / max(float(avg_delay), 1.0))))
        
        strength = (frequency_strength + consistency_strength) / 2.0
        
        if strength < 0.3:  # Minimum strength threshold
            return None
        
        # Create representative events
        representative_cause_event: CausalEvent = CausalEvent(
            event_id=f"temporal_{cause_type}_{uuid.uuid4().hex[:8]}",
            event_type=cause_type,
            description=f"Temporal pattern: {cause_type}",
            timestamp=cause_events[0].timestamp,
            agent_id=cause_events[0].agent_id
        )
        
        representative_effect_event: CausalEvent = CausalEvent(
            event_id=f"temporal_{effect_type}_{uuid.uuid4().hex[:8]}",
            event_type=effect_type,
            description=f"Temporal pattern: {effect_type}",
            timestamp=effect_events[0].timestamp,
            agent_id=effect_events[0].agent_id
        )
        
        # Create evidence
        evidence = CausalEvidence(
            evidence_type=EvidenceType.TEMPORAL,
            description=f"Temporal correlation between {cause_type} and {effect_type}",
            strength=strength,
            confidence=min(0.9, strength + 0.1),
            source="temporal_analysis",
            statistical_measures={
                "correlation_count": correlation_count,
                "average_delay_ms": avg_delay,
                "delay_std_ms": delay_std,
                "frequency_strength": frequency_strength,
                "consistency_strength": consistency_strength
            }
        )
        
        relationship = CausalRelationship(
            relationship_id=f"temporal_{cause_type}_{effect_type}_{uuid.uuid4().hex[:8]}",
            cause_event=representative_cause_event,
            effect_event=representative_effect_event,
            causal_type=CausalType.DIRECT,
            strength=strength,
            confidence=evidence.confidence,
            delay_ms=int(avg_delay),
            evidence=[evidence],
            statistical_significance=float(1.0 - (delay_std / max(float(avg_delay), 1.0)))
        )
        
        return relationship
    
    async def discover_relationships(
        self, 
        agent_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[str]] = None,
        min_confidence: float = 0.5,
        max_relationships: int = 100
    ) -> List[CausalRelationship]:
        """Discover causal relationships for a specific agent with filtering."""
        try:
            # In a real implementation, this would query events from the database
            # and then call the existing discover_relationships method
            # For now, return empty list
            return []
        except Exception as e:
            self.logger.error(f"Failed to discover relationships for agent {agent_id}: {e}")
            raise

    async def _detect_statistical_relationships(self, 
                                              event_groups: Dict[str, List[TemporalEvent]]) -> List[CausalRelationship]:
        """Detect statistical causal relationships using advanced methods."""
        relationships: List[CausalRelationship] = []
        
        # Convert events to time series for statistical analysis
        time_series_data = self._events_to_time_series(event_groups)
        
        if time_series_data is None or len(time_series_data) < self.min_observations:
            return relationships
        
        # Test all pairs of variables for Granger causality
        columns = time_series_data.columns
        for cause_col in columns:
            for effect_col in columns:
                if cause_col != effect_col:
                    try:
                        granger_result = await self.test_granger_causality(
                            time_series_data, cause_col, effect_col
                        )
                        
                        if granger_result.is_causal and granger_result.confidence > 0.7:
                            relationship = self._granger_to_causal_relationship(granger_result)
                            if relationship:
                                relationships.append(relationship)
                    except Exception as e:
                        self.logger.debug(f"Granger test failed for {cause_col} -> {effect_col}: {e}")
        
        return relationships
    
    def _events_to_time_series(self, event_groups: Dict[str, List[TemporalEvent]]) -> Optional[pd.DataFrame]:
        """Convert event groups to time series data for statistical analysis."""
        if not event_groups:
            return None
        
        # Find time range
        all_events = []
        for events in event_groups.values():
            all_events.extend(events)
        
        if not all_events:
            return None
        
        min_time = min(event.timestamp for event in all_events)
        max_time = max(event.timestamp for event in all_events)
        
        # Create time bins (e.g., 1-minute intervals)
        time_range = max_time - min_time
        if time_range.total_seconds() < 60:  # Less than 1 minute of data
            return None
        
        # Create time index
        freq = '1min'  # 1-minute frequency
        time_index = pd.date_range(start=min_time, end=max_time, freq=freq)
        
        # Create DataFrame
        df = pd.DataFrame(index=time_index)
        
        # Count events in each time bin for each event type
        for event_type, events in event_groups.items():
            counts = []
            for timestamp in time_index:
                count = sum(1 for event in events 
                           if timestamp <= event.timestamp < timestamp + pd.Timedelta(minutes=1))
                counts.append(count)
            df[event_type] = counts
        
        return df if len(df) >= self.min_observations else None
    
    def _granger_to_causal_relationship(self, granger_result: GrangerResult) -> Optional[CausalRelationship]:
        """Convert Granger test result to CausalRelationship."""
        if not granger_result.is_causal:
            return None
        
        # Create representative events
        cause_event = CausalEvent(
            event_id=f"granger_{granger_result.cause_variable}_{uuid.uuid4().hex[:8]}",
            event_type=granger_result.cause_variable,
            description=f"Granger cause: {granger_result.cause_variable}",
            timestamp=datetime.utcnow(),
            agent_id="statistical_analysis"
        )
        
        effect_event = CausalEvent(
            event_id=f"granger_{granger_result.effect_variable}_{uuid.uuid4().hex[:8]}",
            event_type=granger_result.effect_variable,
            description=f"Granger effect: {granger_result.effect_variable}",
            timestamp=datetime.utcnow(),
            agent_id="statistical_analysis"
        )
        
        # Create evidence
        evidence = CausalEvidence(
            evidence_type=EvidenceType.STATISTICAL,
            description=f"Granger causality test: {granger_result.cause_variable} -> {granger_result.effect_variable}",
            strength=granger_result.confidence,
            confidence=granger_result.confidence,
            source="granger_causality_test",
            statistical_measures={
                "p_values": granger_result.p_values,
                "f_statistics": granger_result.f_statistics,
                "optimal_lag": granger_result.optimal_lag
            }
        )
        
        relationship = CausalRelationship(
            relationship_id=f"granger_{granger_result.cause_variable}_{granger_result.effect_variable}_{uuid.uuid4().hex[:8]}",
            cause_event=cause_event,
            effect_event=effect_event,
            causal_type=CausalType.DIRECT,
            strength=granger_result.confidence,
            confidence=granger_result.confidence,
            delay_ms=granger_result.optimal_lag * 60000,  # Convert lag to milliseconds (assuming 1-minute intervals)
            evidence=[evidence],
            statistical_significance=1.0 - min(granger_result.p_values.values()) if granger_result.p_values else 0.0,
            causal_mechanism="Granger causality"
        )
        
        return relationship