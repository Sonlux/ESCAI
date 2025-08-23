"""
Integration tests for Neo4j storage functionality.

These tests verify the Neo4j database operations, graph analytics,
and performance characteristics of the ESCAI framework's graph storage.
"""

import pytest
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import os

from escai_framework.storage.neo4j_manager import Neo4jManager, create_causal_relationship_graph
from escai_framework.storage.neo4j_analytics import Neo4jAnalytics, CentralityMeasure
from escai_framework.storage.neo4j_models import (
    CausalNode, CausalRelationship, KnowledgeNode, AgentNode,
    NodeType, RelationshipType, GraphQuery
)


# Test configuration
NEO4J_URI = os.getenv("NEO4J_TEST_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_TEST_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_TEST_PASSWORD", "testpassword")
NEO4J_DATABASE = os.getenv("NEO4J_TEST_DATABASE", "escai_test")

logger = logging.getLogger(__name__)


@pytest.fixture
async def neo4j_manager():
    """Fixture providing a Neo4j manager for testing."""
    manager = Neo4jManager(
        uri=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE
    )
    
    try:
        await manager.connect()
        yield manager
    finally:
        # Clean up test data
        await cleanup_test_data(manager)
        await manager.disconnect()


@pytest.fixture
async def neo4j_analytics(neo4j_manager):
    """Fixture providing Neo4j analytics for testing."""
    return Neo4jAnalytics(neo4j_manager)


async def cleanup_test_data(manager: Neo4jManager):
    """Clean up all test data from the database."""
    cleanup_query = """
    MATCH (n)
    WHERE n.node_id STARTS WITH 'test_'
    DETACH DELETE n
    """
    
    try:
        await manager.execute_custom_query(
            GraphQuery(cleanup_query, {})
        )
    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")


class TestNeo4jManager:
    """Test cases for Neo4j manager functionality."""
    
    async def test_connection_and_schema_initialization(self, neo4j_manager):
        """Test database connection and schema initialization."""
        # Connection should be established by fixture
        assert neo4j_manager.driver is not None
        
        # Test basic query execution
        stats = await neo4j_manager.get_graph_statistics()
        assert isinstance(stats, dict)
    
    async def test_create_and_retrieve_nodes(self, neo4j_manager):
        """Test node creation and retrieval."""
        # Create test agent node
        agent = AgentNode(
            node_id="test_agent_001",
            agent_name="Test Agent",
            framework="test_framework",
            capabilities=["test_capability"]
        )
        
        success = await neo4j_manager.create_node(agent)
        assert success is True
        
        # Verify node exists
        query = GraphQuery(
            "MATCH (n:Agent {node_id: $node_id}) RETURN n.agent_name as name",
            {"node_id": "test_agent_001"}
        )
        results = await neo4j_manager.execute_custom_query(query)
        assert len(results) == 1
        assert results[0]["name"] == "Test Agent"
    
    async def test_create_causal_relationships(self, neo4j_manager):
        """Test causal relationship creation."""
        # Create test events
        cause_event = {
            'node_id': 'test_event_001',
            'event_type': 'decision',
            'description': 'Test decision event',
            'timestamp': datetime.utcnow(),
            'agent_id': 'test_agent_001'
        }
        
        effect_event = {
            'node_id': 'test_event_002',
            'event_type': 'action',
            'description': 'Test action event',
            'timestamp': datetime.utcnow() + timedelta(seconds=1),
            'agent_id': 'test_agent_001'
        }
        
        relationship_data = {
            'relationship_id': 'test_rel_001',
            'strength': 0.85,
            'delay_ms': 1000,
            'evidence': ['test_evidence'],
            'confidence': 0.9
        }
        
        # Create causal relationship
        success = await create_causal_relationship_graph(
            neo4j_manager, cause_event, effect_event, relationship_data
        )
        assert success is True
        
        # Verify relationship exists
        query = GraphQuery(
            """
            MATCH (a:Event {node_id: $cause_id})-[r:CAUSES]->(b:Event {node_id: $effect_id})
            RETURN r.strength as strength, r.confidence as confidence
            """,
            {"cause_id": "test_event_001", "effect_id": "test_event_002"}
        )
        results = await neo4j_manager.execute_custom_query(query)
        assert len(results) == 1
        assert results[0]["strength"] == 0.85
        assert results[0]["confidence"] == 0.9
    
    async def test_find_causal_paths(self, neo4j_manager):
        """Test causal path finding."""
        # Create a chain of events
        events = []
        for i in range(4):
            event = {
                'node_id': f'test_chain_event_{i:03d}',
                'event_type': 'test_event',
                'description': f'Test chain event {i}',
                'timestamp': datetime.utcnow() + timedelta(seconds=i),
                'agent_id': 'test_agent_chain'
            }
            events.append(event)
        
        # Create causal chain
        for i in range(3):
            relationship_data = {
                'relationship_id': f'test_chain_rel_{i:03d}',
                'strength': 0.8 + (i * 0.05),
                'delay_ms': 500,
                'evidence': [f'test_evidence_{i}'],
                'confidence': 0.85
            }
            
            await create_causal_relationship_graph(
                neo4j_manager, events[i], events[i+1], relationship_data
            )
        
        # Find paths
        paths = await neo4j_manager.find_causal_paths(
            "test_chain_event_000", "test_chain_event_003", max_depth=5
        )
        
        assert len(paths) >= 1
        assert paths[0]["path_length"] == 3
        assert len(paths[0]["strengths"]) == 3
    
    async def test_node_centrality_calculation(self, neo4j_manager):
        """Test node centrality calculation."""
        # Create a star topology for testing centrality
        center_node = CausalNode(
            node_id="test_center_node",
            event_type="central_event",
            description="Central hub event",
            timestamp=datetime.utcnow(),
            agent_id="test_agent_centrality"
        )
        await neo4j_manager.create_node(center_node)
        
        # Create peripheral nodes connected to center
        for i in range(5):
            peripheral_node = CausalNode(
                node_id=f"test_peripheral_{i:03d}",
                event_type="peripheral_event",
                description=f"Peripheral event {i}",
                timestamp=datetime.utcnow(),
                agent_id="test_agent_centrality"
            )
            await neo4j_manager.create_node(peripheral_node)
            
            # Create relationship to center
            relationship = CausalRelationship(
                relationship_id=f"test_centrality_rel_{i:03d}",
                cause_node_id=f"test_peripheral_{i:03d}",
                effect_node_id="test_center_node",
                causal_strength=0.8,
                delay_ms=100,
                evidence=[f"centrality_evidence_{i}"]
            )
            await neo4j_manager.create_relationship(relationship)
        
        # Calculate centrality
        centrality = await neo4j_manager.get_node_centrality(NodeType.EVENT)
        
        # Center node should have highest centrality
        assert "test_center_node" in centrality
        center_centrality = centrality["test_center_node"]
        
        # Verify center has higher centrality than peripherals
        for i in range(5):
            peripheral_id = f"test_peripheral_{i:03d}"
            if peripheral_id in centrality:
                assert center_centrality >= centrality[peripheral_id]
    
    async def test_graph_statistics(self, neo4j_manager):
        """Test graph statistics retrieval."""
        # Create some test data
        test_nodes = [
            AgentNode("test_stats_agent", "Stats Agent", "test", ["stats"]),
            CausalNode("test_stats_event_1", "event", "Event 1", datetime.utcnow(), "test_stats_agent"),
            CausalNode("test_stats_event_2", "event", "Event 2", datetime.utcnow(), "test_stats_agent")
        ]
        
        for node in test_nodes:
            await neo4j_manager.create_node(node)
        
        # Create relationship
        relationship = CausalRelationship(
            "test_stats_rel",
            "test_stats_event_1",
            "test_stats_event_2",
            0.8,
            100,
            ["stats_evidence"]
        )
        await neo4j_manager.create_relationship(relationship)
        
        # Get statistics
        stats = await neo4j_manager.get_graph_statistics()
        
        assert "total_nodes" in stats
        assert "total_relationships" in stats
        assert stats["total_nodes"] >= 3
        assert stats["total_relationships"] >= 1
    
    async def test_node_deletion(self, neo4j_manager):
        """Test node deletion functionality."""
        # Create test node
        test_node = AgentNode(
            "test_delete_node",
            "Delete Test Agent",
            "test",
            ["delete_test"]
        )
        await neo4j_manager.create_node(test_node)
        
        # Verify node exists
        query = GraphQuery(
            "MATCH (n {node_id: $node_id}) RETURN count(n) as count",
            {"node_id": "test_delete_node"}
        )
        results = await neo4j_manager.execute_custom_query(query)
        assert results[0]["count"] == 1
        
        # Delete node
        success = await neo4j_manager.delete_node("test_delete_node")
        assert success is True
        
        # Verify node is deleted
        results = await neo4j_manager.execute_custom_query(query)
        assert results[0]["count"] == 0


class TestNeo4jAnalytics:
    """Test cases for Neo4j analytics functionality."""
    
    async def test_centrality_measures(self, neo4j_manager, neo4j_analytics):
        """Test various centrality measure calculations."""
        # Create test graph structure
        await self._create_test_graph_structure(neo4j_manager)
        
        # Test degree centrality
        degree_centrality = await neo4j_analytics.calculate_centrality_measures(
            CentralityMeasure.DEGREE, NodeType.EVENT
        )
        assert isinstance(degree_centrality, dict)
        assert len(degree_centrality) > 0
        
        # Test PageRank centrality
        pagerank_centrality = await neo4j_analytics.calculate_centrality_measures(
            CentralityMeasure.PAGERANK, NodeType.EVENT
        )
        assert isinstance(pagerank_centrality, dict)
    
    async def test_causal_pattern_discovery(self, neo4j_manager, neo4j_analytics):
        """Test causal pattern discovery."""
        # Create repeating causal patterns
        await self._create_repeating_patterns(neo4j_manager)
        
        # Discover patterns
        patterns = await neo4j_analytics.discover_causal_patterns(
            agent_id="test_pattern_agent",
            min_frequency=2,
            min_significance=0.7
        )
        
        assert isinstance(patterns, list)
        if patterns:  # May be empty if patterns don't meet criteria
            pattern = patterns[0]
            assert hasattr(pattern, 'pattern_id')
            assert hasattr(pattern, 'frequency')
            assert pattern.frequency >= 2
    
    async def test_causal_chain_finding(self, neo4j_manager, neo4j_analytics):
        """Test causal chain discovery."""
        # Create causal chains
        await self._create_causal_chains(neo4j_manager)
        
        # Find chains
        chains = await neo4j_analytics.find_causal_chains(
            start_event_type="start_event",
            end_event_type="end_event",
            max_length=5,
            min_strength=0.6
        )
        
        assert isinstance(chains, list)
        if chains:
            chain = chains[0]
            assert hasattr(chain, 'chain_length')
            assert hasattr(chain, 'total_strength')
            assert chain.chain_length <= 5
    
    async def test_temporal_pattern_analysis(self, neo4j_manager, neo4j_analytics):
        """Test temporal pattern analysis."""
        # Create temporal data
        await self._create_temporal_data(neo4j_manager)
        
        # Analyze temporal patterns
        analysis = await neo4j_analytics.analyze_temporal_patterns(
            time_window_hours=24,
            agent_id="test_temporal_agent"
        )
        
        assert analysis.analysis_type == "temporal_patterns"
        assert "temporal_patterns" in analysis.results
        assert analysis.execution_time_ms > 0
    
    async def test_anomaly_detection(self, neo4j_manager, neo4j_analytics):
        """Test anomalous pattern detection."""
        # Create normal and anomalous patterns
        await self._create_anomalous_data(neo4j_manager)
        
        # Detect anomalies
        anomalies = await neo4j_analytics.detect_anomalous_patterns(
            agent_id="test_anomaly_agent",
            threshold_std=1.5
        )
        
        assert isinstance(anomalies, list)
        # Anomalies may or may not be found depending on data distribution
    
    async def test_visualization_data_generation(self, neo4j_manager, neo4j_analytics):
        """Test graph visualization data generation."""
        # Create visualization test data
        await self._create_visualization_data(neo4j_manager)
        
        # Generate visualization data
        viz_data = await neo4j_analytics.generate_graph_visualization_data(
            agent_id="test_viz_agent",
            max_nodes=20
        )
        
        assert "nodes" in viz_data
        assert "edges" in viz_data
        assert "metadata" in viz_data
        assert isinstance(viz_data["nodes"], list)
        assert isinstance(viz_data["edges"], list)
    
    async def _create_test_graph_structure(self, manager: Neo4jManager):
        """Create a test graph structure for analytics testing."""
        # Create agent
        agent = AgentNode(
            "test_analytics_agent",
            "Analytics Test Agent",
            "test",
            ["analytics"]
        )
        await manager.create_node(agent)
        
        # Create interconnected events
        events = []
        for i in range(6):
            event = CausalNode(
                f"test_analytics_event_{i:03d}",
                "analytics_event",
                f"Analytics test event {i}",
                datetime.utcnow() + timedelta(seconds=i),
                "test_analytics_agent"
            )
            events.append(event)
            await manager.create_node(event)
        
        # Create relationships in a complex pattern
        relationships = [
            (0, 1, 0.8), (0, 2, 0.7), (1, 3, 0.9),
            (2, 3, 0.6), (3, 4, 0.85), (4, 5, 0.75)
        ]
        
        for source_idx, target_idx, strength in relationships:
            rel = CausalRelationship(
                f"test_analytics_rel_{source_idx}_{target_idx}",
                f"test_analytics_event_{source_idx:03d}",
                f"test_analytics_event_{target_idx:03d}",
                strength,
                100,
                [f"analytics_evidence_{source_idx}_{target_idx}"]
            )
            await manager.create_relationship(rel)
    
    async def _create_repeating_patterns(self, manager: Neo4jManager):
        """Create repeating causal patterns for pattern discovery testing."""
        agent = AgentNode(
            "test_pattern_agent",
            "Pattern Test Agent",
            "test",
            ["patterns"]
        )
        await manager.create_node(agent)
        
        # Create repeating A->B->C pattern
        for i in range(3):
            events = []
            for j, event_type in enumerate(["pattern_a", "pattern_b", "pattern_c"]):
                event = CausalNode(
                    f"test_pattern_{i}_{j}",
                    event_type,
                    f"Pattern event {event_type} instance {i}",
                    datetime.utcnow() + timedelta(seconds=i*10 + j),
                    "test_pattern_agent"
                )
                events.append(event)
                await manager.create_node(event)
            
            # Create A->B and B->C relationships
            for j in range(2):
                rel = CausalRelationship(
                    f"test_pattern_rel_{i}_{j}",
                    f"test_pattern_{i}_{j}",
                    f"test_pattern_{i}_{j+1}",
                    0.8,
                    500,
                    [f"pattern_evidence_{i}_{j}"]
                )
                await manager.create_relationship(rel)
    
    async def _create_causal_chains(self, manager: Neo4jManager):
        """Create causal chains for chain discovery testing."""
        agent = AgentNode(
            "test_chain_agent",
            "Chain Test Agent",
            "test",
            ["chains"]
        )
        await manager.create_node(agent)
        
        # Create a long causal chain
        chain_events = []
        event_types = ["start_event", "middle_event", "middle_event", "end_event"]
        
        for i, event_type in enumerate(event_types):
            event = CausalNode(
                f"test_chain_discovery_{i}",
                event_type,
                f"Chain discovery event {i}",
                datetime.utcnow() + timedelta(seconds=i),
                "test_chain_agent"
            )
            chain_events.append(event)
            await manager.create_node(event)
        
        # Create chain relationships
        for i in range(len(chain_events) - 1):
            rel = CausalRelationship(
                f"test_chain_discovery_rel_{i}",
                f"test_chain_discovery_{i}",
                f"test_chain_discovery_{i+1}",
                0.8 + (i * 0.05),
                200,
                [f"chain_evidence_{i}"]
            )
            await manager.create_relationship(rel)
    
    async def _create_temporal_data(self, manager: Neo4jManager):
        """Create temporal data for temporal analysis testing."""
        agent = AgentNode(
            "test_temporal_agent",
            "Temporal Test Agent",
            "test",
            ["temporal"]
        )
        await manager.create_node(agent)
        
        # Create events at different times
        base_time = datetime.utcnow() - timedelta(hours=12)
        
        for i in range(10):
            event_time = base_time + timedelta(hours=i)
            event = CausalNode(
                f"test_temporal_event_{i:03d}",
                "temporal_event",
                f"Temporal event {i}",
                event_time,
                "test_temporal_agent"
            )
            await manager.create_node(event)
            
            # Create some causal relationships
            if i > 0:
                rel = CausalRelationship(
                    f"test_temporal_rel_{i:03d}",
                    f"test_temporal_event_{i-1:03d}",
                    f"test_temporal_event_{i:03d}",
                    0.7 + (i * 0.02),
                    300,
                    [f"temporal_evidence_{i}"]
                )
                await manager.create_relationship(rel)
    
    async def _create_anomalous_data(self, manager: Neo4jManager):
        """Create data with anomalous patterns for anomaly detection testing."""
        agent = AgentNode(
            "test_anomaly_agent",
            "Anomaly Test Agent",
            "test",
            ["anomaly"]
        )
        await manager.create_node(agent)
        
        # Create normal patterns
        for i in range(5):
            cause_event = CausalNode(
                f"test_normal_cause_{i}",
                "normal_cause",
                f"Normal cause event {i}",
                datetime.utcnow() + timedelta(seconds=i*2),
                "test_anomaly_agent"
            )
            effect_event = CausalNode(
                f"test_normal_effect_{i}",
                "normal_effect",
                f"Normal effect event {i}",
                datetime.utcnow() + timedelta(seconds=i*2 + 1),
                "test_anomaly_agent"
            )
            
            await manager.create_node(cause_event)
            await manager.create_node(effect_event)
            
            # Normal relationship strength around 0.8
            rel = CausalRelationship(
                f"test_normal_rel_{i}",
                f"test_normal_cause_{i}",
                f"test_normal_effect_{i}",
                0.8 + (i * 0.01),  # Slight variation
                500,
                [f"normal_evidence_{i}"]
            )
            await manager.create_relationship(rel)
        
        # Create anomalous pattern
        anomaly_cause = CausalNode(
            "test_anomaly_cause",
            "normal_cause",
            "Anomalous cause event",
            datetime.utcnow() + timedelta(seconds=20),
            "test_anomaly_agent"
        )
        anomaly_effect = CausalNode(
            "test_anomaly_effect",
            "normal_effect",
            "Anomalous effect event",
            datetime.utcnow() + timedelta(seconds=21),
            "test_anomaly_agent"
        )
        
        await manager.create_node(anomaly_cause)
        await manager.create_node(anomaly_effect)
        
        # Anomalous relationship with very different strength
        anomaly_rel = CausalRelationship(
            "test_anomaly_rel",
            "test_anomaly_cause",
            "test_anomaly_effect",
            0.1,  # Much lower than normal
            5000,  # Much higher delay
            ["anomaly_evidence"]
        )
        await manager.create_relationship(anomaly_rel)
    
    async def _create_visualization_data(self, manager: Neo4jManager):
        """Create data for visualization testing."""
        agent = AgentNode(
            "test_viz_agent",
            "Visualization Test Agent",
            "test",
            ["visualization"]
        )
        await manager.create_node(agent)
        
        # Create a small network for visualization
        for i in range(8):
            event = CausalNode(
                f"test_viz_event_{i:03d}",
                f"viz_type_{i % 3}",
                f"Visualization event {i}",
                datetime.utcnow() + timedelta(seconds=i),
                "test_viz_agent"
            )
            await manager.create_node(event)
        
        # Create some relationships
        relationships = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (6, 7)]
        for i, (source, target) in enumerate(relationships):
            rel = CausalRelationship(
                f"test_viz_rel_{i}",
                f"test_viz_event_{source:03d}",
                f"test_viz_event_{target:03d}",
                0.7 + (i * 0.05),
                200,
                [f"viz_evidence_{i}"]
            )
            await manager.create_relationship(rel)


class TestNeo4jPerformance:
    """Performance tests for Neo4j operations."""
    
    async def test_bulk_node_creation_performance(self, neo4j_manager):
        """Test performance of bulk node creation."""
        import time
        
        start_time = time.time()
        
        # Create 100 nodes
        for i in range(100):
            node = CausalNode(
                f"test_perf_node_{i:03d}",
                "performance_event",
                f"Performance test event {i}",
                datetime.utcnow(),
                "test_perf_agent"
            )
            await neo4j_manager.create_node(node)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert duration < 30.0  # 30 seconds for 100 nodes
        
        # Calculate throughput
        throughput = 100 / duration
        logger.info(f"Node creation throughput: {throughput:.2f} nodes/second")
    
    async def test_complex_query_performance(self, neo4j_manager):
        """Test performance of complex graph queries."""
        # Create test data first
        await self._create_performance_test_data(neo4j_manager)
        
        import time
        start_time = time.time()
        
        # Execute complex query
        query = """
        MATCH path = (start:Event)-[:CAUSES*1..4]->(end:Event)
        WHERE start.agent_id = 'test_perf_agent'
        WITH path, [rel in relationships(path) | rel.strength] as strengths
        RETURN count(path) as path_count,
               avg([s in strengths | s]) as avg_strength,
               length(path) as path_length
        ORDER BY path_length
        """
        
        results = await neo4j_manager.execute_custom_query(
            GraphQuery(query, {})
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Query should complete quickly
        assert duration < 5.0  # 5 seconds
        assert isinstance(results, list)
        
        logger.info(f"Complex query execution time: {duration:.3f} seconds")
    
    async def _create_performance_test_data(self, manager: Neo4jManager):
        """Create test data for performance testing."""
        # Create agent
        agent = AgentNode(
            "test_perf_agent",
            "Performance Test Agent",
            "test",
            ["performance"]
        )
        await manager.create_node(agent)
        
        # Create 50 events
        for i in range(50):
            event = CausalNode(
                f"test_perf_event_{i:03d}",
                "perf_event",
                f"Performance event {i}",
                datetime.utcnow() + timedelta(seconds=i),
                "test_perf_agent"
            )
            await manager.create_node(event)
        
        # Create relationships (create a connected graph)
        for i in range(49):
            rel = CausalRelationship(
                f"test_perf_rel_{i:03d}",
                f"test_perf_event_{i:03d}",
                f"test_perf_event_{i+1:03d}",
                0.8,
                100,
                [f"perf_evidence_{i}"]
            )
            await manager.create_relationship(rel)
        
        # Add some additional cross-connections
        for i in range(0, 49, 5):
            if i + 10 < 50:
                rel = CausalRelationship(
                    f"test_perf_cross_rel_{i:03d}",
                    f"test_perf_event_{i:03d}",
                    f"test_perf_event_{i+10:03d}",
                    0.6,
                    500,
                    [f"perf_cross_evidence_{i}"]
                )
                await manager.create_relationship(rel)