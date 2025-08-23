"""
Neo4j Storage Example for ESCAI Framework.

This example demonstrates how to use Neo4j for storing and analyzing
causal relationships and knowledge graphs in the ESCAI framework.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

from escai_framework.storage.neo4j_manager import Neo4jManager, create_causal_relationship_graph
from escai_framework.storage.neo4j_analytics import Neo4jAnalytics, CentralityMeasure
from escai_framework.storage.neo4j_models import (
    CausalNode, CausalRelationship, KnowledgeNode, AgentNode,
    NodeType, RelationshipType
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def setup_neo4j_connection() -> Neo4jManager:
    """Set up Neo4j connection with example configuration."""
    # Note: Update these connection details for your Neo4j instance
    manager = Neo4jManager(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password",
        database="escai"
    )
    
    await manager.connect()
    return manager


async def create_sample_causal_graph(manager: Neo4jManager) -> None:
    """Create sample causal relationships for demonstration."""
    logger.info("Creating sample causal graph...")
    
    # Sample agent
    agent = AgentNode(
        node_id="agent_001",
        agent_name="LangChain Assistant",
        framework="langchain",
        capabilities=["reasoning", "tool_use", "memory"]
    )
    await manager.create_node(agent)
    
    # Sample events and causal relationships
    events_data = [
        {
            'node_id': 'event_001',
            'event_type': 'decision',
            'description': 'Agent decides to use search tool',
            'timestamp': datetime.utcnow() - timedelta(minutes=10),
            'agent_id': 'agent_001'
        },
        {
            'node_id': 'event_002',
            'event_type': 'action',
            'description': 'Agent executes search query',
            'timestamp': datetime.utcnow() - timedelta(minutes=9),
            'agent_id': 'agent_001'
        },
        {
            'node_id': 'event_003',
            'event_type': 'outcome',
            'description': 'Search returns relevant results',
            'timestamp': datetime.utcnow() - timedelta(minutes=8),
            'agent_id': 'agent_001'
        },
        {
            'node_id': 'event_004',
            'event_type': 'decision',
            'description': 'Agent decides to synthesize information',
            'timestamp': datetime.utcnow() - timedelta(minutes=7),
            'agent_id': 'agent_001'
        },
        {
            'node_id': 'event_005',
            'event_type': 'outcome',
            'description': 'Agent provides comprehensive answer',
            'timestamp': datetime.utcnow() - timedelta(minutes=5),
            'agent_id': 'agent_001'
        }
    ]
    
    # Create causal relationships
    causal_relationships = [
        {
            'cause': 'event_001',
            'effect': 'event_002',
            'relationship_id': 'rel_001',
            'strength': 0.9,
            'delay_ms': 500,
            'evidence': ['log_entry_1', 'trace_data_1'],
            'confidence': 0.95
        },
        {
            'cause': 'event_002',
            'effect': 'event_003',
            'relationship_id': 'rel_002',
            'strength': 0.8,
            'delay_ms': 1200,
            'evidence': ['api_response', 'timing_data'],
            'confidence': 0.9
        },
        {
            'cause': 'event_003',
            'effect': 'event_004',
            'relationship_id': 'rel_003',
            'strength': 0.85,
            'delay_ms': 300,
            'evidence': ['reasoning_trace', 'context_analysis'],
            'confidence': 0.88
        },
        {
            'cause': 'event_004',
            'effect': 'event_005',
            'relationship_id': 'rel_004',
            'strength': 0.92,
            'delay_ms': 2000,
            'evidence': ['synthesis_log', 'output_generation'],
            'confidence': 0.93
        }
    ]
    
    # Create the causal graph
    for i, rel_data in enumerate(causal_relationships):
        cause_event = next(e for e in events_data if e['node_id'] == rel_data['cause'])
        effect_event = next(e for e in events_data if e['node_id'] == rel_data['effect'])
        
        success = await create_causal_relationship_graph(
            manager, cause_event, effect_event, rel_data
        )
        
        if success:
            logger.info(f"Created causal relationship: {rel_data['cause']} -> {rel_data['effect']}")
        else:
            logger.error(f"Failed to create relationship {i}")
    
    # Create knowledge nodes
    knowledge_nodes = [
        KnowledgeNode(
            node_id="knowledge_001",
            knowledge_type="factual",
            content="Search tools are effective for information retrieval",
            confidence_level=0.9,
            agent_id="agent_001"
        ),
        KnowledgeNode(
            node_id="knowledge_002",
            knowledge_type="procedural",
            content="Synthesis improves answer quality",
            confidence_level=0.85,
            agent_id="agent_001"
        )
    ]
    
    for knowledge in knowledge_nodes:
        await manager.create_node(knowledge)
        logger.info(f"Created knowledge node: {knowledge.node_id}")


async def demonstrate_graph_analytics(manager: Neo4jManager) -> None:
    """Demonstrate graph analytics capabilities."""
    logger.info("Demonstrating graph analytics...")
    
    analytics = Neo4jAnalytics(manager)
    
    # Calculate centrality measures
    logger.info("Calculating degree centrality...")
    degree_centrality = await analytics.calculate_centrality_measures(
        CentralityMeasure.DEGREE, NodeType.EVENT
    )
    logger.info(f"Degree centrality results: {degree_centrality}")
    
    # Discover causal patterns
    logger.info("Discovering causal patterns...")
    patterns = await analytics.discover_causal_patterns(
        agent_id="agent_001", min_frequency=1, min_significance=0.7
    )
    for pattern in patterns:
        logger.info(f"Found pattern: {pattern.description} (frequency: {pattern.frequency})")
    
    # Find causal chains
    logger.info("Finding causal chains...")
    chains = await analytics.find_causal_chains(
        start_event_type="decision", max_length=4, min_strength=0.7
    )
    for chain in chains:
        logger.info(f"Causal chain: {chain.start_event} -> {chain.end_event} "
                   f"(length: {chain.chain_length}, strength: {chain.total_strength:.2f})")
    
    # Analyze temporal patterns
    logger.info("Analyzing temporal patterns...")
    temporal_analysis = await analytics.analyze_temporal_patterns(
        time_window_hours=24, agent_id="agent_001"
    )
    logger.info(f"Temporal analysis completed in {temporal_analysis.execution_time_ms:.2f}ms")
    
    # Detect anomalous patterns
    logger.info("Detecting anomalous patterns...")
    anomalies = await analytics.detect_anomalous_patterns(agent_id="agent_001")
    for anomaly in anomalies:
        logger.info(f"Anomaly detected: {anomaly['cause_type']} -> {anomaly['effect_type']} "
                   f"(z-score: {anomaly['strength_z_score']:.2f})")


async def demonstrate_graph_queries(manager: Neo4jManager) -> None:
    """Demonstrate custom graph queries."""
    logger.info("Demonstrating custom graph queries...")
    
    # Find causal paths
    logger.info("Finding causal paths...")
    paths = await manager.find_causal_paths("event_001", "event_005", max_depth=5)
    for i, path in enumerate(paths):
        logger.info(f"Path {i+1}: length={path['path_length']}, "
                   f"avg_strength={path['average_strength']:.2f}")
    
    # Get node centrality
    logger.info("Getting node centrality...")
    centrality = await manager.get_node_centrality(NodeType.EVENT)
    for node_id, degree in list(centrality.items())[:5]:
        logger.info(f"Node {node_id}: degree={degree}")
    
    # Analyze causal patterns
    logger.info("Analyzing causal patterns...")
    analysis = await manager.analyze_causal_patterns(agent_id="agent_001")
    logger.info(f"Analysis type: {analysis.analysis_type}")
    logger.info(f"Execution time: {analysis.execution_time_ms:.2f}ms")
    logger.info(f"Nodes: {analysis.node_count}, Relationships: {analysis.relationship_count}")


async def demonstrate_visualization_data(manager: Neo4jManager) -> None:
    """Demonstrate graph visualization data generation."""
    logger.info("Generating visualization data...")
    
    analytics = Neo4jAnalytics(manager)
    viz_data = await analytics.generate_graph_visualization_data(
        agent_id="agent_001", max_nodes=50
    )
    
    logger.info(f"Visualization data generated:")
    logger.info(f"  Nodes: {len(viz_data['nodes'])}")
    logger.info(f"  Edges: {len(viz_data['edges'])}")
    
    # Print sample node and edge data
    if viz_data['nodes']:
        sample_node = viz_data['nodes'][0]
        logger.info(f"  Sample node: {sample_node['id']} ({sample_node['type']})")
    
    if viz_data['edges']:
        sample_edge = viz_data['edges'][0]
        logger.info(f"  Sample edge: {sample_edge['source']} -> {sample_edge['target']} "
                   f"(strength: {sample_edge['strength']:.2f})")


async def demonstrate_graph_statistics(manager: Neo4jManager) -> None:
    """Demonstrate graph statistics retrieval."""
    logger.info("Getting graph statistics...")
    
    stats = await manager.get_graph_statistics()
    logger.info(f"Graph statistics:")
    logger.info(f"  Total nodes: {stats.get('total_nodes', 0)}")
    logger.info(f"  Total relationships: {stats.get('total_relationships', 0)}")
    logger.info(f"  Node types: {stats.get('node_types', 0)}")
    logger.info(f"  Relationship types: {stats.get('relationship_types', 0)}")


async def cleanup_sample_data(manager: Neo4jManager) -> None:
    """Clean up sample data."""
    logger.info("Cleaning up sample data...")
    
    # Delete all nodes and relationships for the sample agent
    query = """
    MATCH (n {agent_id: 'agent_001'})
    DETACH DELETE n
    """
    
    try:
        await manager.execute_custom_query(
            type('GraphQuery', (), {'query': query, 'parameters': {}})()
        )
        
        # Also delete the agent node
        await manager.delete_node("agent_001")
        
        logger.info("Sample data cleaned up successfully")
    except Exception as e:
        logger.error(f"Failed to clean up sample data: {e}")


async def main():
    """Main example function."""
    logger.info("Starting Neo4j Storage Example...")
    
    try:
        # Set up connection
        manager = await setup_neo4j_connection()
        
        # Create sample data
        await create_sample_causal_graph(manager)
        
        # Demonstrate analytics
        await demonstrate_graph_analytics(manager)
        
        # Demonstrate queries
        await demonstrate_graph_queries(manager)
        
        # Demonstrate visualization
        await demonstrate_visualization_data(manager)
        
        # Show statistics
        await demonstrate_graph_statistics(manager)
        
        # Clean up
        await cleanup_sample_data(manager)
        
        logger.info("Neo4j example completed successfully!")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise
    finally:
        # Ensure connection is closed
        if 'manager' in locals():
            await manager.disconnect()


if __name__ == "__main__":
    asyncio.run(main())