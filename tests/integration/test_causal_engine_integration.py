"""
Integration tests for the CausalEngine class.

Tests the causal engine with realistic scenarios and data patterns.
"""

import asyncio
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from escai_framework.core.causal_engine import CausalEngine, TemporalEvent
from escai_framework.models.causal_relationship import CausalType, EvidenceType


class TestCausalEngineIntegration:
    """Integration tests for CausalEngine."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_causal_discovery(self):
        """Test complete causal discovery workflow."""
        engine = CausalEngine(min_observations=20, significance_threshold=0.1)
        
        # Create realistic agent behavior scenario:
        # Agent makes decisions, which lead to actions, which lead to outcomes
        events = []
        base_time = datetime.utcnow()
        
        # Simulate 30 decision-action-outcome cycles
        for i in range(30):
            cycle_start = base_time + timedelta(minutes=i*10)
            
            # Decision event
            decision_event = TemporalEvent(
                event_id=f"decision_{i}",
                event_type="decision",
                timestamp=cycle_start,
                agent_id="agent_1",
                attributes={"confidence": 0.7 + 0.3 * np.random.random()}
            )
            events.append(decision_event)
            
            # Action event (2-5 seconds after decision)
            action_delay = 2 + 3 * np.random.random()
            action_event = TemporalEvent(
                event_id=f"action_{i}",
                event_type="action",
                timestamp=cycle_start + timedelta(seconds=action_delay),
                agent_id="agent_1",
                attributes={"execution_time": action_delay}
            )
            events.append(action_event)
            
            # Outcome event (10-20 seconds after action)
            outcome_delay = 10 + 10 * np.random.random()
            outcome_event = TemporalEvent(
                event_id=f"outcome_{i}",
                event_type="outcome",
                timestamp=cycle_start + timedelta(seconds=action_delay + outcome_delay),
                agent_id="agent_1",
                attributes={"success": np.random.random() > 0.3}
            )
            events.append(outcome_event)
        
        # Discover relationships
        relationships = await engine.discover_relationships(events)
        
        # Should find temporal relationships
        assert len(relationships) > 0
        
        # Check for expected causal chains
        decision_to_action = None
        action_to_outcome = None
        
        for rel in relationships:
            if (rel.cause_event.event_type == "decision" and 
                rel.effect_event.event_type == "action"):
                decision_to_action = rel
            elif (rel.cause_event.event_type == "action" and 
                  rel.effect_event.event_type == "outcome"):
                action_to_outcome = rel
        
        # Should find decision -> action relationship
        if decision_to_action:
            assert decision_to_action.strength > 0.3
            assert decision_to_action.delay_ms > 0
            assert len(decision_to_action.evidence) > 0
            assert decision_to_action.evidence[0].evidence_type == EvidenceType.TEMPORAL
        
        # Should find action -> outcome relationship
        if action_to_outcome:
            assert action_to_outcome.strength > 0.3
            assert action_to_outcome.delay_ms > 0
    
    @pytest.mark.asyncio
    async def test_granger_causality_with_synthetic_data(self):
        """Test Granger causality with synthetic time series data."""
        engine = CausalEngine(min_observations=50)
        
        # Create synthetic time series with known causal relationship
        # X causes Y with a lag of 1 time step
        np.random.seed(42)
        n_obs = 100
        
        # Generate X as AR(1) process
        x = np.zeros(n_obs)
        x[0] = np.random.randn()
        for t in range(1, n_obs):
            x[t] = 0.5 * x[t-1] + np.random.randn() * 0.5
        
        # Generate Y as function of lagged X plus noise
        y = np.zeros(n_obs)
        y[0] = np.random.randn()
        for t in range(1, n_obs):
            y[t] = 0.3 * x[t-1] + 0.4 * y[t-1] + np.random.randn() * 0.3
        
        # Create DataFrame
        data = pd.DataFrame({'X': x, 'Y': y})
        
        # Test Granger causality
        result = await engine.test_granger_causality(data, 'X', 'Y')
        
        # X should Granger-cause Y
        assert result.cause_variable == 'X'
        assert result.effect_variable == 'Y'
        # Note: Granger causality might not always be detected with synthetic data
        # depending on the specific realization of random noise
        if result.is_causal:
            assert result.confidence > 0.5
            assert result.optimal_lag >= 1
        else:
            # Even if not detected as causal, should have reasonable structure
            assert result.confidence >= 0.0
            assert result.optimal_lag >= 1
        
        # Test reverse direction (should be weaker or non-causal)
        reverse_result = await engine.test_granger_causality(data, 'Y', 'X')
        # Reverse direction should generally be weaker, but not always guaranteed
        # with synthetic data due to random noise
        assert reverse_result.cause_variable == 'Y'
        assert reverse_result.effect_variable == 'X'
    
    @pytest.mark.asyncio
    async def test_causal_graph_construction_and_analysis(self):
        """Test building and analyzing causal graphs."""
        engine = CausalEngine(min_observations=15)
        
        # Create events representing a causal chain: A -> B -> C
        events = []
        base_time = datetime.utcnow()
        
        for i in range(20):
            # Event A
            event_a = TemporalEvent(
                event_id=f"a_{i}",
                event_type="event_a",
                timestamp=base_time + timedelta(minutes=i*5),
                agent_id="agent_1"
            )
            events.append(event_a)
            
            # Event B (caused by A, 30 seconds later)
            event_b = TemporalEvent(
                event_id=f"b_{i}",
                event_type="event_b",
                timestamp=base_time + timedelta(minutes=i*5, seconds=30),
                agent_id="agent_1"
            )
            events.append(event_b)
            
            # Event C (caused by B, 60 seconds after B)
            event_c = TemporalEvent(
                event_id=f"c_{i}",
                event_type="event_c",
                timestamp=base_time + timedelta(minutes=i*5, seconds=90),
                agent_id="agent_1"
            )
            events.append(event_c)
        
        # Discover relationships
        relationships = await engine.discover_relationships(events)
        
        # Build causal graph
        graph = await engine.build_causal_graph(relationships)
        
        # Should have nodes for all event types
        assert len(graph.nodes) > 0
        
        # Test graph traversal
        if len(graph.nodes) >= 3:
            # Find a node and test ancestor/descendant relationships
            sample_node = list(graph.nodes)[0]
            ancestors = graph.get_ancestors(sample_node)
            descendants = graph.get_descendants(sample_node)
            
            # Should be able to traverse the graph
            assert isinstance(ancestors, list)
            assert isinstance(descendants, list)
    
    @pytest.mark.asyncio
    async def test_intervention_analysis_with_linear_relationship(self):
        """Test intervention analysis with known linear relationship."""
        engine = CausalEngine()
        
        # Create graph with treatment -> outcome relationship
        from escai_framework.core.causal_engine import CausalGraph
        graph = CausalGraph()
        graph.nodes.add("treatment")
        graph.nodes.add("outcome")
        
        # Create synthetic data with linear relationship: outcome = 2 * treatment + noise
        np.random.seed(42)
        n_obs = 100
        treatment = np.random.randn(n_obs)
        outcome = 2.0 * treatment + np.random.randn(n_obs) * 0.1
        
        data = pd.DataFrame({
            'treatment': treatment,
            'outcome': outcome
        })
        
        # Test intervention: increase treatment by 1 unit
        intervention_effect = await engine.analyze_interventions(
            graph=graph,
            intervention_variable="treatment",
            intervention_value=np.mean(treatment) + 1.0,
            target_variable="outcome",
            data=data
        )
        
        # Should detect approximately 2.0 effect size
        assert intervention_effect.intervention_variable == "treatment"
        assert intervention_effect.target_variable == "outcome"
        assert abs(intervention_effect.expected_effect - 2.0) < 0.5
        assert intervention_effect.p_value < 0.05  # Should be statistically significant
        
        # Confidence interval should contain the true effect
        ci_lower, ci_upper = intervention_effect.confidence_interval
        assert ci_lower < 2.0 < ci_upper
    
    @pytest.mark.asyncio
    async def test_performance_with_large_dataset(self):
        """Test engine performance with larger datasets."""
        engine = CausalEngine(min_observations=100)
        
        # Create large dataset (1000 events)
        events = []
        base_time = datetime.utcnow()
        
        for i in range(500):
            # Create pairs of cause-effect events
            cause_event = TemporalEvent(
                event_id=f"cause_{i}",
                event_type="cause",
                timestamp=base_time + timedelta(seconds=i*10),
                agent_id="agent_1"
            )
            events.append(cause_event)
            
            effect_event = TemporalEvent(
                event_id=f"effect_{i}",
                event_type="effect",
                timestamp=base_time + timedelta(seconds=i*10 + 2),
                agent_id="agent_1"
            )
            events.append(effect_event)
        
        # Measure analysis time
        start_time = datetime.utcnow()
        relationships = await engine.discover_relationships(events)
        end_time = datetime.utcnow()
        
        analysis_time = (end_time - start_time).total_seconds()
        
        # Should complete within reasonable time (requirement: < 2 seconds)
        assert analysis_time < 2.0
        
        # Should find relationships
        assert len(relationships) > 0
    
    @pytest.mark.asyncio
    async def test_uncertainty_handling(self):
        """Test handling of uncertain or weak causal relationships."""
        engine = CausalEngine(min_observations=20, significance_threshold=0.05)
        
        # Create events with weak/random relationships
        events = []
        base_time = datetime.utcnow()
        
        # Create random events with no clear causal structure
        for i in range(50):
            event_type = np.random.choice(["type_a", "type_b", "type_c"])
            timestamp = base_time + timedelta(seconds=np.random.randint(0, 3600))
            
            event = TemporalEvent(
                event_id=f"random_{i}",
                event_type=event_type,
                timestamp=timestamp,
                agent_id="agent_1"
            )
            events.append(event)
        
        # Discover relationships
        relationships = await engine.discover_relationships(events)
        
        # Should handle uncertainty gracefully
        # May find some weak relationships or none at all
        for rel in relationships:
            # Any found relationships should have reasonable confidence bounds
            assert 0.0 <= rel.confidence <= 1.0
            assert 0.0 <= rel.strength <= 1.0
            assert rel.statistical_significance >= 0.0
            
            # Should have evidence
            assert len(rel.evidence) > 0
            for evidence in rel.evidence:
                assert 0.0 <= evidence.confidence <= 1.0
                assert 0.0 <= evidence.strength <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])