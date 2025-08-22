"""
Example demonstrating the ESCAI framework's causal inference capabilities.

This example shows how to use the CausalEngine to discover causal relationships
in agent behavior, perform Granger causality testing, and analyze interventions.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from escai_framework.core.causal_engine import CausalEngine, TemporalEvent


async def demonstrate_temporal_causality():
    """Demonstrate temporal causality detection."""
    print("=== Temporal Causality Detection ===")
    
    # Initialize the causal engine
    engine = CausalEngine(min_observations=20, significance_threshold=0.1)
    
    # Create a scenario where agent decisions lead to actions
    events = []
    base_time = datetime.utcnow()
    
    print("Creating synthetic agent behavior data...")
    
    # Simulate 25 decision-action cycles
    for i in range(25):
        cycle_start = base_time + timedelta(minutes=i*5)
        
        # Decision event
        decision_event = TemporalEvent(
            event_id=f"decision_{i}",
            event_type="decision",
            timestamp=cycle_start,
            agent_id="agent_1",
            attributes={"confidence": 0.6 + 0.4 * np.random.random()}
        )
        events.append(decision_event)
        
        # Action event (1-3 seconds after decision)
        action_delay = 1 + 2 * np.random.random()
        action_event = TemporalEvent(
            event_id=f"action_{i}",
            event_type="action",
            timestamp=cycle_start + timedelta(seconds=action_delay),
            agent_id="agent_1",
            attributes={"execution_time": action_delay}
        )
        events.append(action_event)
    
    print(f"Created {len(events)} events")
    
    # Discover causal relationships
    print("Analyzing temporal patterns...")
    relationships = await engine.discover_relationships(events)
    
    print(f"Found {len(relationships)} causal relationships:")
    for rel in relationships:
        print(f"  {rel.cause_event.event_type} -> {rel.effect_event.event_type}")
        print(f"    Strength: {rel.strength:.3f}")
        print(f"    Confidence: {rel.confidence:.3f}")
        print(f"    Delay: {rel.delay_ms:.0f}ms")
        print(f"    Evidence: {len(rel.evidence)} pieces")
        print()


async def demonstrate_granger_causality():
    """Demonstrate Granger causality testing."""
    print("=== Granger Causality Testing ===")
    
    engine = CausalEngine(min_observations=50)
    
    # Create synthetic time series with known causal relationship
    print("Creating synthetic time series data...")
    np.random.seed(42)
    n_obs = 100
    
    # Generate X as AR(1) process
    x = np.zeros(n_obs)
    x[0] = np.random.randn()
    for t in range(1, n_obs):
        x[t] = 0.6 * x[t-1] + np.random.randn() * 0.4
    
    # Generate Y as function of lagged X plus its own lag
    y = np.zeros(n_obs)
    y[0] = np.random.randn()
    for t in range(1, n_obs):
        y[t] = 0.4 * x[t-1] + 0.3 * y[t-1] + np.random.randn() * 0.3
    
    # Create DataFrame
    data = pd.DataFrame({'X': x, 'Y': y})
    print(f"Created time series with {len(data)} observations")
    
    # Test Granger causality X -> Y
    print("Testing Granger causality: X -> Y")
    result_xy = await engine.test_granger_causality(data, 'X', 'Y')
    
    print(f"  Cause: {result_xy.cause_variable}")
    print(f"  Effect: {result_xy.effect_variable}")
    print(f"  Is Causal: {result_xy.is_causal}")
    print(f"  Confidence: {result_xy.confidence:.3f}")
    print(f"  Optimal Lag: {result_xy.optimal_lag}")
    print(f"  P-values: {result_xy.p_values}")
    
    # Test reverse direction Y -> X
    print("\nTesting reverse direction: Y -> X")
    result_yx = await engine.test_granger_causality(data, 'Y', 'X')
    
    print(f"  Is Causal: {result_yx.is_causal}")
    print(f"  Confidence: {result_yx.confidence:.3f}")
    
    print(f"\nConclusion: X -> Y causality is {'stronger' if result_xy.confidence > result_yx.confidence else 'weaker'} than Y -> X")


async def demonstrate_intervention_analysis():
    """Demonstrate intervention effect analysis."""
    print("=== Intervention Analysis ===")
    
    engine = CausalEngine()
    
    # Create synthetic data with linear relationship
    print("Creating synthetic intervention data...")
    np.random.seed(42)
    n_obs = 100
    
    # Treatment variable
    treatment = np.random.randn(n_obs)
    
    # Outcome = 1.5 * treatment + noise
    outcome = 1.5 * treatment + np.random.randn(n_obs) * 0.2
    
    data = pd.DataFrame({
        'treatment': treatment,
        'outcome': outcome
    })
    
    print(f"True causal effect: 1.5 (outcome = 1.5 * treatment + noise)")
    
    # Create simple causal graph
    from escai_framework.core.causal_engine import CausalGraph
    graph = CausalGraph()
    graph.nodes.add("treatment")
    graph.nodes.add("outcome")
    
    # Analyze intervention: increase treatment by 1 unit
    current_treatment = np.mean(treatment)
    intervention_value = current_treatment + 1.0
    
    print(f"Analyzing intervention: treatment {current_treatment:.2f} -> {intervention_value:.2f}")
    
    intervention_effect = await engine.analyze_interventions(
        graph=graph,
        intervention_variable="treatment",
        intervention_value=intervention_value,
        target_variable="outcome",
        data=data
    )
    
    print(f"  Expected Effect: {intervention_effect.expected_effect:.3f}")
    print(f"  Confidence Interval: ({intervention_effect.confidence_interval[0]:.3f}, {intervention_effect.confidence_interval[1]:.3f})")
    print(f"  P-value: {intervention_effect.p_value:.6f}")
    print(f"  Statistical Significance: {'Yes' if intervention_effect.p_value < 0.05 else 'No'}")


async def demonstrate_causal_graph():
    """Demonstrate causal graph construction and analysis."""
    print("=== Causal Graph Analysis ===")
    
    engine = CausalEngine(min_observations=15)
    
    # Create events representing a causal chain: Planning -> Execution -> Evaluation
    events = []
    base_time = datetime.utcnow()
    
    print("Creating causal chain: Planning -> Execution -> Evaluation")
    
    for i in range(20):
        # Planning event
        planning_event = TemporalEvent(
            event_id=f"planning_{i}",
            event_type="planning",
            timestamp=base_time + timedelta(minutes=i*10),
            agent_id="agent_1"
        )
        events.append(planning_event)
        
        # Execution event (2 minutes after planning)
        execution_event = TemporalEvent(
            event_id=f"execution_{i}",
            event_type="execution",
            timestamp=base_time + timedelta(minutes=i*10 + 2),
            agent_id="agent_1"
        )
        events.append(execution_event)
        
        # Evaluation event (5 minutes after execution)
        evaluation_event = TemporalEvent(
            event_id=f"evaluation_{i}",
            event_type="evaluation",
            timestamp=base_time + timedelta(minutes=i*10 + 7),
            agent_id="agent_1"
        )
        events.append(evaluation_event)
    
    print(f"Created {len(events)} events")
    
    # Discover relationships
    relationships = await engine.discover_relationships(events)
    
    # Build causal graph
    graph = await engine.build_causal_graph(relationships)
    
    print(f"Causal graph contains {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    print("Nodes:", list(graph.nodes))
    
    if graph.edges:
        print("Causal relationships:")
        for cause, effect, strength in graph.edges:
            print(f"  {cause} -> {effect} (strength: {strength:.3f})")
        
        # Analyze graph structure
        if len(graph.nodes) > 0:
            sample_node = list(graph.nodes)[0]
            ancestors = graph.get_ancestors(sample_node)
            descendants = graph.get_descendants(sample_node)
            
            print(f"\nGraph analysis for node '{sample_node}':")
            print(f"  Ancestors: {ancestors}")
            print(f"  Descendants: {descendants}")


async def main():
    """Run all causal analysis demonstrations."""
    print("ESCAI Framework - Causal Analysis Examples")
    print("=" * 50)
    
    try:
        await demonstrate_temporal_causality()
        print("\n" + "=" * 50 + "\n")
        
        await demonstrate_granger_causality()
        print("\n" + "=" * 50 + "\n")
        
        await demonstrate_intervention_analysis()
        print("\n" + "=" * 50 + "\n")
        
        await demonstrate_causal_graph()
        
        print("\n" + "=" * 50)
        print("All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())