"""
Unit tests for the CausalEngine class.

Tests cover temporal causality detection, Granger causality testing,
causal graph construction, and intervention analysis.
"""

import asyncio
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from escai_framework.core.causal_engine import (
    CausalEngine, TemporalEvent, GrangerResult, CausalGraph, InterventionEffect
)
from escai_framework.models.causal_relationship import (
    CausalRelationship, CausalEvent, CausalEvidence, CausalType, EvidenceType
)


class TestTemporalEvent:
    """Test TemporalEvent class."""
    
    def test_temporal_event_creation(self):
        """Test creating a temporal event."""
        timestamp = datetime.utcnow()
        event = TemporalEvent(
            event_id="test_event_1",
            event_type="decision",
            timestamp=timestamp,
            agent_id="agent_1",
            attributes={"confidence": 0.8}
        )
        
        assert event.event_id == "test_event_1"
        assert event.event_type == "decision"
        assert event.timestamp == timestamp
        assert event.agent_id == "agent_1"
        assert event.attributes["confidence"] == 0.8
    
    def test_temporal_event_to_dict(self):
        """Test converting temporal event to dictionary."""
        timestamp = datetime.utcnow()
        event = TemporalEvent(
            event_id="test_event_1",
            event_type="decision",
            timestamp=timestamp,
            agent_id="agent_1",
            attributes={"confidence": 0.8}
        )
        
        event_dict = event.to_dict()
        assert event_dict["event_id"] == "test_event_1"
        assert event_dict["event_type"] == "decision"
        assert event_dict["timestamp"] == timestamp.isoformat()
        assert event_dict["agent_id"] == "agent_1"
        assert event_dict["attributes"]["confidence"] == 0.8


class TestGrangerResult:
    """Test GrangerResult class."""
    
    def test_granger_result_creation(self):
        """Test creating a Granger result."""
        result = GrangerResult(
            cause_variable="cause",
            effect_variable="effect",
            p_values={1: 0.01, 2: 0.05},
            f_statistics={1: 10.5, 2: 8.2},
            optimal_lag=1,
            is_causal=True,
            confidence=0.99
        )
        
        assert result.cause_variable == "cause"
        assert result.effect_variable == "effect"
        assert result.p_values[1] == 0.01
        assert result.f_statistics[1] == 10.5
        assert result.optimal_lag == 1
        assert result.is_causal is True
        assert result.confidence == 0.99
    
    def test_granger_result_to_dict(self):
        """Test converting Granger result to dictionary."""
        result = GrangerResult(
            cause_variable="cause",
            effect_variable="effect",
            p_values={1: 0.01},
            f_statistics={1: 10.5},
            optimal_lag=1,
            is_causal=True,
            confidence=0.99
        )
        
        result_dict = result.to_dict()
        assert result_dict["cause_variable"] == "cause"
        assert result_dict["effect_variable"] == "effect"
        assert result_dict["is_causal"] is True


class TestCausalGraph:
    """Test CausalGraph class."""
    
    def test_causal_graph_creation(self):
        """Test creating an empty causal graph."""
        graph = CausalGraph()
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
        assert len(graph.relationships) == 0
    
    def test_add_relationship(self):
        """Test adding a relationship to the graph."""
        graph = CausalGraph()
        
        # Create test relationship
        cause_event = CausalEvent(
            event_id="cause_1",
            event_type="decision",
            description="Test cause",
            timestamp=datetime.utcnow(),
            agent_id="agent_1"
        )
        
        effect_event = CausalEvent(
            event_id="effect_1",
            event_type="action",
            description="Test effect",
            timestamp=datetime.utcnow(),
            agent_id="agent_1"
        )
        
        relationship = CausalRelationship(
            relationship_id="rel_1",
            cause_event=cause_event,
            effect_event=effect_event,
            causal_type=CausalType.DIRECT,
            strength=0.8,
            confidence=0.9,
            delay_ms=1000
        )
        
        graph.add_relationship(relationship)
        
        assert "cause_1" in graph.nodes
        assert "effect_1" in graph.nodes
        assert len(graph.edges) == 1
        assert graph.edges[0] == ("cause_1", "effect_1", 0.8)
        assert ("cause_1", "effect_1") in graph.relationships
    
    def test_get_ancestors(self):
        """Test getting ancestor nodes."""
        graph = CausalGraph()
        
        # Create chain: A -> B -> C
        events = []
        relationships = []
        
        for i, event_id in enumerate(["A", "B", "C"]):
            event = CausalEvent(
                event_id=event_id,
                event_type="test",
                description=f"Event {event_id}",
                timestamp=datetime.utcnow() + timedelta(seconds=i),
                agent_id="agent_1"
            )
            events.append(event)
        
        # A -> B
        rel1 = CausalRelationship(
            relationship_id="rel_1",
            cause_event=events[0],
            effect_event=events[1],
            causal_type=CausalType.DIRECT,
            strength=0.8,
            confidence=0.9,
            delay_ms=1000
        )
        
        # B -> C
        rel2 = CausalRelationship(
            relationship_id="rel_2",
            cause_event=events[1],
            effect_event=events[2],
            causal_type=CausalType.DIRECT,
            strength=0.7,
            confidence=0.8,
            delay_ms=1000
        )
        
        graph.add_relationship(rel1)
        graph.add_relationship(rel2)
        
        ancestors_c = graph.get_ancestors("C")
        assert "B" in ancestors_c
        assert "A" in ancestors_c
        
        ancestors_b = graph.get_ancestors("B")
        assert "A" in ancestors_b
        assert "C" not in ancestors_b
    
    def test_get_descendants(self):
        """Test getting descendant nodes."""
        graph = CausalGraph()
        
        # Create the same chain: A -> B -> C
        events = []
        
        for i, event_id in enumerate(["A", "B", "C"]):
            event = CausalEvent(
                event_id=event_id,
                event_type="test",
                description=f"Event {event_id}",
                timestamp=datetime.utcnow() + timedelta(seconds=i),
                agent_id="agent_1"
            )
            events.append(event)
        
        # A -> B
        rel1 = CausalRelationship(
            relationship_id="rel_1",
            cause_event=events[0],
            effect_event=events[1],
            causal_type=CausalType.DIRECT,
            strength=0.8,
            confidence=0.9,
            delay_ms=1000
        )
        
        # B -> C
        rel2 = CausalRelationship(
            relationship_id="rel_2",
            cause_event=events[1],
            effect_event=events[2],
            causal_type=CausalType.DIRECT,
            strength=0.7,
            confidence=0.8,
            delay_ms=1000
        )
        
        graph.add_relationship(rel1)
        graph.add_relationship(rel2)
        
        descendants_a = graph.get_descendants("A")
        assert "B" in descendants_a
        assert "C" in descendants_a
        
        descendants_b = graph.get_descendants("B")
        assert "C" in descendants_b
        assert "A" not in descendants_b


class TestInterventionEffect:
    """Test InterventionEffect class."""
    
    def test_intervention_effect_creation(self):
        """Test creating an intervention effect."""
        effect = InterventionEffect(
            intervention_variable="treatment",
            target_variable="outcome",
            intervention_value=1.0,
            expected_effect=0.5,
            confidence_interval=(0.2, 0.8),
            p_value=0.01
        )
        
        assert effect.intervention_variable == "treatment"
        assert effect.target_variable == "outcome"
        assert effect.intervention_value == 1.0
        assert effect.expected_effect == 0.5
        assert effect.confidence_interval == (0.2, 0.8)
        assert effect.p_value == 0.01
    
    def test_intervention_effect_to_dict(self):
        """Test converting intervention effect to dictionary."""
        effect = InterventionEffect(
            intervention_variable="treatment",
            target_variable="outcome",
            intervention_value=1.0,
            expected_effect=0.5,
            confidence_interval=(0.2, 0.8),
            p_value=0.01
        )
        
        effect_dict = effect.to_dict()
        assert effect_dict["intervention_variable"] == "treatment"
        assert effect_dict["target_variable"] == "outcome"
        assert effect_dict["expected_effect"] == 0.5


class TestCausalEngine:
    """Test CausalEngine class."""
    
    def test_causal_engine_initialization(self):
        """Test initializing the causal engine."""
        engine = CausalEngine(
            significance_threshold=0.01,
            max_lag=5,
            min_observations=30
        )
        
        assert engine.significance_threshold == 0.01
        assert engine.max_lag == 5
        assert engine.min_observations == 30
    
    def test_causal_engine_default_initialization(self):
        """Test initializing with default parameters."""
        engine = CausalEngine()
        
        assert engine.significance_threshold == 0.05
        assert engine.max_lag == 10
        assert engine.min_observations == 50
    
    @pytest.mark.asyncio
    async def test_discover_relationships_insufficient_events(self):
        """Test discovering relationships with insufficient events."""
        engine = CausalEngine(min_observations=50)
        
        # Create few events
        events = []
        for i in range(10):
            event = TemporalEvent(
                event_id=f"event_{i}",
                event_type="test",
                timestamp=datetime.utcnow() + timedelta(seconds=i),
                agent_id="agent_1"
            )
            events.append(event)
        
        relationships = await engine.discover_relationships_from_events(events)
        assert len(relationships) == 0
    
    @pytest.mark.asyncio
    async def test_discover_relationships_temporal_pattern(self):
        """Test discovering temporal relationships."""
        engine = CausalEngine(min_observations=10)
        
        # Create events with temporal pattern: decision -> action
        events = []
        base_time = datetime.utcnow()
        
        # Create decision events
        for i in range(10):
            decision_event = TemporalEvent(
                event_id=f"decision_{i}",
                event_type="decision",
                timestamp=base_time + timedelta(seconds=i*60),
                agent_id="agent_1"
            )
            events.append(decision_event)
            
            # Create corresponding action event 5 seconds later
            action_event = TemporalEvent(
                event_id=f"action_{i}",
                event_type="action",
                timestamp=base_time + timedelta(seconds=i*60 + 5),
                agent_id="agent_1"
            )
            events.append(action_event)
        
        relationships = await engine.discover_relationships_from_events(events)
        
        # Should find temporal relationship between decision and action
        assert len(relationships) > 0
        
        # Check if we found the expected relationship
        decision_to_action = None
        for rel in relationships:
            if (rel.cause_event.event_type == "decision" and 
                rel.effect_event.event_type == "action"):
                decision_to_action = rel
                break
        
        assert decision_to_action is not None
        assert decision_to_action.strength > 0.3
        assert decision_to_action.delay_ms > 0
    
    @pytest.mark.asyncio
    async def test_test_granger_causality_missing_columns(self):
        """Test Granger causality with missing columns."""
        engine = CausalEngine()
        
        # Create test data
        data = pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100)
        })
        
        with pytest.raises(ValueError, match="Columns.*not found"):
            await engine.test_granger_causality(data, "missing_col", "y")
    
    @pytest.mark.asyncio
    async def test_test_granger_causality_insufficient_data(self):
        """Test Granger causality with insufficient data."""
        engine = CausalEngine(min_observations=100)
        
        # Create small dataset
        data = pd.DataFrame({
            'x': np.random.randn(10),
            'y': np.random.randn(10)
        })
        
        with pytest.raises(ValueError, match="Insufficient observations"):
            await engine.test_granger_causality(data, "x", "y")
    
    @pytest.mark.asyncio
    @patch('escai_framework.core.causal_engine.STATSMODELS_AVAILABLE', True)
    @patch('escai_framework.core.causal_engine.grangercausalitytests')
    async def test_test_granger_causality_success(self, mock_granger):
        """Test successful Granger causality test."""
        engine = CausalEngine(min_observations=50)
        
        # Mock Granger test results
        mock_granger.return_value = {
            1: [{'ssr_ftest': (10.5, 0.01)}],
            2: [{'ssr_ftest': (8.2, 0.05)}]
        }
        
        # Create test data
        data = pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100)
        })
        
        result = await engine.test_granger_causality(data, "x", "y")
        
        assert result.cause_variable == "x"
        assert result.effect_variable == "y"
        assert result.is_causal is True
        assert result.optimal_lag == 1
        assert result.confidence > 0.9
        assert 1 in result.p_values
        assert 1 in result.f_statistics
    
    @pytest.mark.asyncio
    @patch('escai_framework.core.causal_engine.STATSMODELS_AVAILABLE', False)
    async def test_test_granger_causality_no_statsmodels(self):
        """Test Granger causality without statsmodels."""
        engine = CausalEngine()
        
        data = pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100)
        })
        
        with pytest.raises(RuntimeError, match="statsmodels is required"):
            await engine.test_granger_causality(data, "x", "y")
    
    @pytest.mark.asyncio
    async def test_build_causal_graph(self):
        """Test building a causal graph from relationships."""
        engine = CausalEngine()
        
        # Create test relationships
        relationships = []
        
        for i in range(3):
            cause_event = CausalEvent(
                event_id=f"cause_{i}",
                event_type="decision",
                description=f"Cause {i}",
                timestamp=datetime.utcnow(),
                agent_id="agent_1"
            )
            
            effect_event = CausalEvent(
                event_id=f"effect_{i}",
                event_type="action",
                description=f"Effect {i}",
                timestamp=datetime.utcnow(),
                agent_id="agent_1"
            )
            
            relationship = CausalRelationship(
                relationship_id=f"rel_{i}",
                cause_event=cause_event,
                effect_event=effect_event,
                causal_type=CausalType.DIRECT,
                strength=0.8,
                confidence=0.9 if i < 2 else 0.3,  # Last one has low confidence
                delay_ms=1000
            )
            relationships.append(relationship)
        
        graph = await engine.build_causal_graph(relationships)
        
        # Should only include high-confidence relationships
        assert len(graph.nodes) == 4  # 2 relationships * 2 nodes each
        assert len(graph.edges) == 2  # Only high-confidence relationships
    
    @pytest.mark.asyncio
    async def test_analyze_interventions_missing_variables(self):
        """Test intervention analysis with missing variables."""
        engine = CausalEngine()
        graph = CausalGraph()
        data = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        
        effect = await engine.analyze_interventions(
            graph, "missing_var", 1.0, "y", data
        )
        
        assert effect.expected_effect == 0.0
        assert effect.confidence_interval == (0.0, 0.0)
        assert effect.p_value == 1.0
    
    @pytest.mark.asyncio
    async def test_analyze_interventions_success(self):
        """Test successful intervention analysis."""
        engine = CausalEngine()
        
        # Create graph with nodes
        graph = CausalGraph()
        graph.nodes.add("x")
        graph.nodes.add("y")
        
        # Create data with linear relationship
        np.random.seed(42)
        x_values = np.random.randn(100)
        y_values = 2 * x_values + np.random.randn(100) * 0.1  # y = 2x + noise
        
        data = pd.DataFrame({'x': x_values, 'y': y_values})
        
        effect = await engine.analyze_interventions(
            graph, "x", 1.0, "y", data
        )
        
        assert effect.intervention_variable == "x"
        assert effect.target_variable == "y"
        assert effect.intervention_value == 1.0
        assert abs(effect.expected_effect - 2.0) < 0.5  # Should be close to 2
        assert effect.p_value < 0.05  # Should be significant
    
    def test_group_events(self):
        """Test grouping events by type."""
        engine = CausalEngine()
        
        events = []
        for i in range(6):
            event_type = "decision" if i % 2 == 0 else "action"
            event = TemporalEvent(
                event_id=f"event_{i}",
                event_type=event_type,
                timestamp=datetime.utcnow() + timedelta(seconds=i),
                agent_id="agent_1"
            )
            events.append(event)
        
        groups = engine._group_events(events)
        
        assert "decision" in groups
        assert "action" in groups
        assert len(groups["decision"]) == 3
        assert len(groups["action"]) == 3
    
    def test_events_to_time_series_empty(self):
        """Test converting empty events to time series."""
        engine = CausalEngine()
        
        result = engine._events_to_time_series({})
        assert result is None
    
    def test_events_to_time_series_insufficient_time_range(self):
        """Test converting events with insufficient time range."""
        engine = CausalEngine()
        
        base_time = datetime.utcnow()
        events = [
            TemporalEvent("e1", "decision", base_time, "agent_1"),
            TemporalEvent("e2", "action", base_time + timedelta(seconds=30), "agent_1")
        ]
        
        groups = {"decision": [events[0]], "action": [events[1]]}
        result = engine._events_to_time_series(groups)
        
        assert result is None
    
    def test_events_to_time_series_success(self):
        """Test successful conversion of events to time series."""
        engine = CausalEngine(min_observations=10)
        
        base_time = datetime.utcnow()
        events = []
        
        # Create events over 20 minutes
        for i in range(20):
            decision_event = TemporalEvent(
                f"decision_{i}", "decision", 
                base_time + timedelta(minutes=i), "agent_1"
            )
            action_event = TemporalEvent(
                f"action_{i}", "action", 
                base_time + timedelta(minutes=i, seconds=30), "agent_1"
            )
            events.extend([decision_event, action_event])
        
        groups = engine._group_events(events)
        result = engine._events_to_time_series(groups)
        
        assert result is not None
        assert "decision" in result.columns
        assert "action" in result.columns
        assert len(result) >= 10
    
    def test_granger_to_causal_relationship_non_causal(self):
        """Test converting non-causal Granger result."""
        engine = CausalEngine()
        
        granger_result = GrangerResult(
            cause_variable="x",
            effect_variable="y",
            p_values={1: 0.8},
            f_statistics={1: 0.5},
            optimal_lag=1,
            is_causal=False,
            confidence=0.2
        )
        
        result = engine._granger_to_causal_relationship(granger_result)
        assert result is None
    
    def test_granger_to_causal_relationship_success(self):
        """Test successful conversion of Granger result to causal relationship."""
        engine = CausalEngine()
        
        granger_result = GrangerResult(
            cause_variable="x",
            effect_variable="y",
            p_values={1: 0.01, 2: 0.05},
            f_statistics={1: 10.5, 2: 8.2},
            optimal_lag=1,
            is_causal=True,
            confidence=0.99
        )
        
        result = engine._granger_to_causal_relationship(granger_result)
        
        assert result is not None
        assert result.cause_event.event_type == "x"
        assert result.effect_event.event_type == "y"
        assert result.strength == 0.99
        assert result.confidence == 0.99
        assert result.causal_mechanism == "Granger causality"
        assert result.validated is True
        assert len(result.evidence) == 1
        assert result.evidence[0].evidence_type == EvidenceType.STATISTICAL


if __name__ == "__main__":
    pytest.main([__file__])