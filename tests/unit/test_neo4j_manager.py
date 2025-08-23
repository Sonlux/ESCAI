"""
Unit tests for Neo4j manager functionality.

These tests verify the Neo4j manager's core functionality using mocks
to avoid requiring an actual Neo4j database connection.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from typing import Dict, Any, List

from escai_framework.storage.neo4j_manager import Neo4jManager, create_causal_relationship_graph
from escai_framework.storage.neo4j_models import (
    CausalNode, CausalRelationship, KnowledgeNode, AgentNode,
    NodeType, RelationshipType, GraphQuery, GraphAnalysisResult
)


def create_mock_manager():
    """Helper function to create a properly mocked Neo4j manager."""
    manager = Neo4jManager("bolt://localhost:7687", "user", "pass")
    
    # Create mock driver and session
    mock_driver = AsyncMock()
    
    # Create a mock session that will be returned by the context manager
    mock_session = AsyncMock()
    
    # Create a proper async context manager
    class MockAsyncContextManager:
        async def __aenter__(self):
            return mock_session
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None
    
    # Make session() return the async context manager
    mock_driver.session.return_value = MockAsyncContextManager()
    
    manager.driver = mock_driver
    # Store reference to mock_session for easy access in tests
    manager._mock_session = mock_session
    return manager


class TestNeo4jManagerInitialization:
    """Test Neo4j manager initialization and connection."""
    
    def test_manager_initialization(self):
        """Test Neo4j manager initialization with parameters."""
        manager = Neo4jManager(
            uri="bolt://localhost:7687",
            username="test_user",
            password="test_password",
            database="test_db",
            max_connection_lifetime=1800
        )
        
        assert manager.uri == "bolt://localhost:7687"
        assert manager.username == "test_user"
        assert manager.password == "test_password"
        assert manager.database == "test_db"
        assert manager.max_connection_lifetime == 1800
        assert manager.driver is None
    
    def test_manager_default_parameters(self):
        """Test Neo4j manager with default parameters."""
        manager = Neo4jManager(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="password"
        )
        
        assert manager.database == "neo4j"
        assert manager.max_connection_lifetime == 3600


class TestNeo4jManagerConnection:
    """Test Neo4j manager connection functionality."""
    
    @pytest.fixture
    def mock_driver(self):
        """Mock Neo4j driver."""
        driver = AsyncMock()
        driver.verify_connectivity = AsyncMock()
        driver.session = AsyncMock()
        driver.close = AsyncMock()
        return driver
    
    @pytest.fixture
    def manager(self):
        """Neo4j manager instance for testing."""
        return Neo4jManager(
            uri="bolt://localhost:7687",
            username="test_user",
            password="test_password"
        )
    
    @patch('escai_framework.storage.neo4j_manager.AsyncGraphDatabase')
    async def test_successful_connection(self, mock_graph_db, manager, mock_driver):
        """Test successful database connection."""
        mock_graph_db.driver.return_value = mock_driver
        
        # Mock session for schema initialization
        mock_session = AsyncMock()
        mock_driver.session.return_value.__aenter__.return_value = mock_session
        mock_session.run = AsyncMock()
        
        await manager.connect()
        
        assert manager.driver == mock_driver
        mock_driver.verify_connectivity.assert_called_once()
        mock_graph_db.driver.assert_called_once_with(
            "bolt://localhost:7687",
            auth=("test_user", "test_password"),
            max_connection_lifetime=3600
        )
    
    @patch('escai_framework.storage.neo4j_manager.AsyncGraphDatabase')
    async def test_connection_failure(self, mock_graph_db, manager):
        """Test connection failure handling."""
        from neo4j.exceptions import ServiceUnavailable
        
        mock_driver = AsyncMock()
        mock_driver.verify_connectivity.side_effect = ServiceUnavailable("Connection failed")
        mock_graph_db.driver.return_value = mock_driver
        
        with pytest.raises(ServiceUnavailable):
            await manager.connect()
    
    async def test_disconnect(self, manager, mock_driver):
        """Test database disconnection."""
        manager.driver = mock_driver
        
        await manager.disconnect()
        
        mock_driver.close.assert_called_once()


class TestNeo4jManagerNodeOperations:
    """Test Neo4j manager node operations."""
    
    @pytest.fixture
    def manager_with_mock_driver(self):
        """Manager with mocked driver."""
        return create_mock_manager()
    
    async def test_create_node_success(self, manager_with_mock_driver):
        """Test successful node creation."""
        manager = manager_with_mock_driver
        
        # Use the stored mock session
        mock_session = manager._mock_session
        mock_result = AsyncMock()
        mock_record = {"node_id": "test_node_001"}
        
        mock_session.run.return_value = mock_result
        mock_result.single.return_value = mock_record
        
        # Create test node
        node = AgentNode(
            node_id="test_node_001",
            agent_name="Test Agent",
            framework="test",
            capabilities=["test"]
        )
        
        success = await manager.create_node(node)
        
        assert success is True
        mock_session.run.assert_called_once()
        
        # Verify the query contains the node type and properties
        call_args = mock_session.run.call_args[0]
        query = call_args[0]
        assert ":Agent" in query
        assert "test_node_001" in query
    
    async def test_create_node_failure(self, manager_with_mock_driver):
        """Test node creation failure."""
        manager = manager_with_mock_driver
        
        # Mock session that raises exception
        mock_session = AsyncMock()
        mock_session.run.side_effect = Exception("Database error")
        
        manager.driver.session.return_value.__aenter__.return_value = mock_session
        
        node = AgentNode("test_node", "Test", "test", [])
        success = await manager.create_node(node)
        
        assert success is False
    
    async def test_create_node_no_result(self, manager_with_mock_driver):
        """Test node creation with no result returned."""
        manager = manager_with_mock_driver
        
        # Mock session with no result
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.single.return_value = None
        
        manager.driver.session.return_value.__aenter__.return_value = mock_session
        mock_session.run.return_value = mock_result
        
        node = AgentNode("test_node", "Test", "test", [])
        success = await manager.create_node(node)
        
        assert success is False


class TestNeo4jManagerRelationshipOperations:
    """Test Neo4j manager relationship operations."""
    
    @pytest.fixture
    def manager_with_mock_driver(self):
        """Manager with mocked driver."""
        return create_mock_manager()
    
    async def test_create_relationship_success(self, manager_with_mock_driver):
        """Test successful relationship creation."""
        manager = manager_with_mock_driver
        
        # Mock session and result
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_record = {"relationship_id": "test_rel_001"}
        
        manager.driver.session.return_value.__aenter__.return_value = mock_session
        mock_session.run.return_value = mock_result
        mock_result.single.return_value = mock_record
        
        # Create test relationship
        relationship = CausalRelationship(
            relationship_id="test_rel_001",
            cause_node_id="cause_001",
            effect_node_id="effect_001",
            causal_strength=0.8,
            delay_ms=500,
            evidence=["test_evidence"]
        )
        
        success = await manager.create_relationship(relationship)
        
        assert success is True
        mock_session.run.assert_called_once()
        
        # Verify the query and parameters
        call_args = mock_session.run.call_args
        query = call_args[0][0]
        parameters = call_args[0][1]
        
        assert ":CAUSES" in query
        assert parameters["source_id"] == "cause_001"
        assert parameters["target_id"] == "effect_001"
    
    async def test_create_relationship_failure(self, manager_with_mock_driver):
        """Test relationship creation failure."""
        manager = manager_with_mock_driver
        
        # Mock session that raises exception
        mock_session = AsyncMock()
        mock_session.run.side_effect = Exception("Database error")
        
        manager.driver.session.return_value.__aenter__.return_value = mock_session
        
        relationship = CausalRelationship(
            "test_rel", "cause", "effect", 0.8, 500, ["evidence"]
        )
        success = await manager.create_relationship(relationship)
        
        assert success is False


class TestNeo4jManagerQueryOperations:
    """Test Neo4j manager query operations."""
    
    @pytest.fixture
    def manager_with_mock_driver(self):
        """Manager with mocked driver."""
        return create_mock_manager()
    
    async def test_find_causal_paths_success(self, manager_with_mock_driver):
        """Test successful causal path finding."""
        manager = manager_with_mock_driver
        
        # Mock session and result
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        
        # Mock result records
        mock_records = [
            {
                "path_length": 2,
                "strengths": [0.8, 0.9],
                "confidences": [0.85, 0.95]
            },
            {
                "path_length": 3,
                "strengths": [0.7, 0.8, 0.6],
                "confidences": [0.8, 0.9, 0.7]
            }
        ]
        
        async def mock_records_iterator():
            for record in mock_records:
                yield record
        
        manager.driver.session.return_value.__aenter__.return_value = mock_session
        mock_session.run.return_value = mock_result
        mock_result.__aiter__ = mock_records_iterator
        
        paths = await manager.find_causal_paths("start_node", "end_node", max_depth=5)
        
        assert len(paths) == 2
        assert paths[0]["path_length"] == 2
        assert paths[0]["average_strength"] == 0.85  # (0.8 + 0.9) / 2
        assert paths[1]["path_length"] == 3
        assert paths[1]["average_confidence"] == pytest.approx(0.8333, rel=1e-3)
    
    async def test_find_causal_paths_empty_result(self, manager_with_mock_driver):
        """Test causal path finding with empty result."""
        manager = manager_with_mock_driver
        
        # Mock session with empty result
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        
        async def empty_iterator():
            return
            yield  # This line will never be reached
        
        manager.driver.session.return_value.__aenter__.return_value = mock_session
        mock_session.run.return_value = mock_result
        mock_result.__aiter__ = empty_iterator
        
        paths = await manager.find_causal_paths("start", "end")
        
        assert paths == []
    
    async def test_get_node_centrality_success(self, manager_with_mock_driver):
        """Test successful node centrality calculation."""
        manager = manager_with_mock_driver
        
        # Mock session and result
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        
        mock_records = [
            {"node_id": "node_001", "degree": 5},
            {"node_id": "node_002", "degree": 3},
            {"node_id": "node_003", "degree": 1}
        ]
        
        async def mock_records_iterator():
            for record in mock_records:
                yield record
        
        manager.driver.session.return_value.__aenter__.return_value = mock_session
        mock_session.run.return_value = mock_result
        mock_result.__aiter__ = mock_records_iterator
        
        centrality = await manager.get_node_centrality(NodeType.EVENT)
        
        assert centrality == {
            "node_001": 5,
            "node_002": 3,
            "node_003": 1
        }
    
    async def test_execute_custom_query_success(self, manager_with_mock_driver):
        """Test successful custom query execution."""
        manager = manager_with_mock_driver
        
        # Mock session and result
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        
        mock_records = [
            {"name": "Agent 1", "count": 10},
            {"name": "Agent 2", "count": 5}
        ]
        
        async def mock_records_iterator():
            for record in mock_records:
                yield record
        
        manager.driver.session.return_value.__aenter__.return_value = mock_session
        mock_session.run.return_value = mock_result
        mock_result.__aiter__ = mock_records_iterator
        
        query = GraphQuery(
            "MATCH (n:Agent) RETURN n.name as name, count(*) as count",
            {}
        )
        
        results = await manager.execute_custom_query(query)
        
        assert len(results) == 2
        assert results[0] == {"name": "Agent 1", "count": 10}
        assert results[1] == {"name": "Agent 2", "count": 5}
    
    async def test_delete_node_success(self, manager_with_mock_driver):
        """Test successful node deletion."""
        manager = manager_with_mock_driver
        
        # Mock session and result
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_record = {"deleted_count": 1}
        
        manager.driver.session.return_value.__aenter__.return_value = mock_session
        mock_session.run.return_value = mock_result
        mock_result.single.return_value = mock_record
        
        success = await manager.delete_node("test_node_001")
        
        assert success is True
        mock_session.run.assert_called_once()
        
        # Verify the query contains DETACH DELETE
        call_args = mock_session.run.call_args[0]
        query = call_args[0]
        assert "DETACH DELETE" in query
    
    async def test_get_graph_statistics_success(self, manager_with_mock_driver):
        """Test successful graph statistics retrieval."""
        manager = manager_with_mock_driver
        
        # Mock session and result
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_record = {
            "total_nodes": 100,
            "total_relationships": 250,
            "node_types": 5,
            "relationship_types": 3
        }
        
        manager.driver.session.return_value.__aenter__.return_value = mock_session
        mock_session.run.return_value = mock_result
        mock_result.single.return_value = mock_record
        
        stats = await manager.get_graph_statistics()
        
        assert stats["total_nodes"] == 100
        assert stats["total_relationships"] == 250
        assert stats["node_types"] == 5
        assert stats["relationship_types"] == 3


class TestNeo4jManagerAnalysisOperations:
    """Test Neo4j manager analysis operations."""
    
    @pytest.fixture
    def manager_with_mock_driver(self):
        """Manager with mocked driver."""
        return create_mock_manager()
    
    async def test_analyze_causal_patterns_success(self, manager_with_mock_driver):
        """Test successful causal pattern analysis."""
        manager = manager_with_mock_driver
        
        # Mock session and multiple query results
        mock_session = AsyncMock()
        
        # Mock results for different queries
        count_result = AsyncMock()
        count_result.single.return_value = {"node_count": 50, "rel_count": 75}
        
        influence_records = [
            {
                "event_id": "event_001",
                "event_type": "decision",
                "outgoing_causes": 5,
                "avg_strength": 0.8
            }
        ]
        
        chains_records = [
            {
                "chain_length": 3,
                "avg_strength": 0.75,
                "frequency": 10
            }
        ]
        
        async def influence_iterator():
            for record in influence_records:
                yield record
        
        async def chains_iterator():
            for record in chains_records:
                yield record
        
        influence_result = AsyncMock()
        influence_result.__aiter__ = influence_iterator
        
        chains_result = AsyncMock()
        chains_result.__aiter__ = chains_iterator
        
        # Configure mock session to return different results for different queries
        mock_session.run.side_effect = [count_result, influence_result, chains_result]
        
        manager.driver.session.return_value.__aenter__.return_value = mock_session
        
        analysis = await manager.analyze_causal_patterns(agent_id="test_agent")
        
        assert analysis.analysis_type == "causal_patterns"
        assert analysis.node_count == 50
        assert analysis.relationship_count == 75
        assert "influential_events" in analysis.results
        assert "causal_chains" in analysis.results
        assert len(analysis.results["influential_events"]) == 1
        assert len(analysis.results["causal_chains"]) == 1
    
    async def test_analyze_causal_patterns_failure(self, manager_with_mock_driver):
        """Test causal pattern analysis failure handling."""
        manager = manager_with_mock_driver
        
        # Mock session that raises exception
        mock_session = AsyncMock()
        mock_session.run.side_effect = Exception("Database error")
        
        manager.driver.session.return_value.__aenter__.return_value = mock_session
        
        analysis = await manager.analyze_causal_patterns()
        
        assert analysis.analysis_type == "causal_patterns"
        assert "error" in analysis.results
        assert analysis.node_count == 0
        assert analysis.relationship_count == 0


class TestCreateCausalRelationshipGraph:
    """Test the convenience function for creating causal relationship graphs."""
    
    @pytest.fixture
    def mock_manager(self):
        """Mock Neo4j manager."""
        manager = AsyncMock(spec=Neo4jManager)
        manager.create_node = AsyncMock(return_value=True)
        manager.create_relationship = AsyncMock(return_value=True)
        return manager
    
    async def test_create_causal_relationship_graph_success(self, mock_manager):
        """Test successful causal relationship graph creation."""
        cause_event = {
            'node_id': 'cause_001',
            'event_type': 'decision',
            'description': 'Agent makes decision',
            'timestamp': datetime.utcnow(),
            'agent_id': 'agent_001'
        }
        
        effect_event = {
            'node_id': 'effect_001',
            'event_type': 'action',
            'description': 'Agent takes action',
            'timestamp': datetime.utcnow() + timedelta(seconds=1),
            'agent_id': 'agent_001'
        }
        
        relationship_data = {
            'relationship_id': 'rel_001',
            'strength': 0.85,
            'delay_ms': 1000,
            'evidence': ['log_entry_1'],
            'confidence': 0.9
        }
        
        success = await create_causal_relationship_graph(
            mock_manager, cause_event, effect_event, relationship_data
        )
        
        assert success is True
        
        # Verify that create_node was called twice (for cause and effect)
        assert mock_manager.create_node.call_count == 2
        
        # Verify that create_relationship was called once
        mock_manager.create_relationship.assert_called_once()
        
        # Verify the relationship parameters
        relationship_call = mock_manager.create_relationship.call_args[0][0]
        assert relationship_call.relationship_id == 'rel_001'
        assert relationship_call.source_node_id == 'cause_001'
        assert relationship_call.target_node_id == 'effect_001'
        assert relationship_call.strength == 0.85
    
    async def test_create_causal_relationship_graph_node_failure(self, mock_manager):
        """Test causal relationship graph creation with node creation failure."""
        # Mock one node creation to fail
        mock_manager.create_node.side_effect = [True, False]  # First succeeds, second fails
        
        cause_event = {
            'node_id': 'cause_002',
            'event_type': 'decision',
            'description': 'Test cause',
            'timestamp': datetime.utcnow(),
            'agent_id': 'agent_002'
        }
        
        effect_event = {
            'node_id': 'effect_002',
            'event_type': 'action',
            'description': 'Test effect',
            'timestamp': datetime.utcnow(),
            'agent_id': 'agent_002'
        }
        
        relationship_data = {
            'relationship_id': 'rel_002',
            'strength': 0.7,
            'delay_ms': 500,
            'evidence': ['evidence'],
            'confidence': 0.8
        }
        
        success = await create_causal_relationship_graph(
            mock_manager, cause_event, effect_event, relationship_data
        )
        
        assert success is False
    
    async def test_create_causal_relationship_graph_relationship_failure(self, mock_manager):
        """Test causal relationship graph creation with relationship creation failure."""
        # Mock nodes to succeed but relationship to fail
        mock_manager.create_node.return_value = True
        mock_manager.create_relationship.return_value = False
        
        cause_event = {
            'node_id': 'cause_003',
            'event_type': 'decision',
            'description': 'Test cause',
            'timestamp': datetime.utcnow(),
            'agent_id': 'agent_003'
        }
        
        effect_event = {
            'node_id': 'effect_003',
            'event_type': 'action',
            'description': 'Test effect',
            'timestamp': datetime.utcnow(),
            'agent_id': 'agent_003'
        }
        
        relationship_data = {
            'relationship_id': 'rel_003',
            'strength': 0.6,
            'delay_ms': 200,
            'evidence': ['test_evidence'],
            'confidence': 0.75
        }
        
        success = await create_causal_relationship_graph(
            mock_manager, cause_event, effect_event, relationship_data
        )
        
        assert success is False


class TestNeo4jManagerErrorHandling:
    """Test error handling in Neo4j manager."""
    
    @pytest.fixture
    def manager_with_mock_driver(self):
        """Manager with mocked driver."""
        return create_mock_manager()
    
    async def test_query_execution_exception_handling(self, manager_with_mock_driver):
        """Test exception handling during query execution."""
        manager = manager_with_mock_driver
        
        # Mock session that raises exception
        mock_session = AsyncMock()
        mock_session.run.side_effect = Exception("Connection lost")
        
        manager.driver.session.return_value.__aenter__.return_value = mock_session
        
        # Test various operations that should handle exceptions gracefully
        paths = await manager.find_causal_paths("start", "end")
        assert paths == []
        
        centrality = await manager.get_node_centrality()
        assert centrality == {}
        
        stats = await manager.get_graph_statistics()
        assert stats == {}
        
        custom_results = await manager.execute_custom_query(
            GraphQuery("MATCH (n) RETURN n", {})
        )
        assert custom_results == []
    
    async def test_session_context_manager_exception(self, manager_with_mock_driver):
        """Test exception handling in session context manager."""
        manager = manager_with_mock_driver
        
        # Mock session context manager to raise exception
        manager.driver.session.side_effect = Exception("Session creation failed")
        
        node = AgentNode("test", "Test", "test", [])
        success = await manager.create_node(node)
        
        assert success is False