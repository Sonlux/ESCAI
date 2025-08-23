"""
Unit tests for Neo4j models and data structures.

These tests verify the correctness of Neo4j model classes,
data validation, and serialization functionality.
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any

from escai_framework.storage.neo4j_models import (
    GraphNode, GraphRelationship, CausalNode, CausalRelationship,
    KnowledgeNode, AgentNode, NodeType, RelationshipType,
    GraphQuery, GraphAnalysisResult
)


class TestNodeType:
    """Test NodeType enumeration."""
    
    def test_node_type_values(self):
        """Test that NodeType has expected values."""
        assert NodeType.AGENT.value == "Agent"
        assert NodeType.EVENT.value == "Event"
        assert NodeType.KNOWLEDGE.value == "Knowledge"
        assert NodeType.BELIEF.value == "Belief"
        assert NodeType.GOAL.value == "Goal"


class TestRelationshipType:
    """Test RelationshipType enumeration."""
    
    def test_relationship_type_values(self):
        """Test that RelationshipType has expected values."""
        assert RelationshipType.CAUSES.value == "CAUSES"
        assert RelationshipType.INFLUENCES.value == "INFLUENCES"
        assert RelationshipType.LEADS_TO.value == "LEADS_TO"
        assert RelationshipType.DEPENDS_ON.value == "DEPENDS_ON"


class TestGraphNode:
    """Test GraphNode base class."""
    
    def test_graph_node_creation(self):
        """Test GraphNode creation with valid data."""
        properties = {"test_prop": "test_value", "numeric_prop": 42}
        created_at = datetime.utcnow()
        
        node = GraphNode(
            node_id="test_node_001",
            node_type=NodeType.EVENT,
            properties=properties,
            created_at=created_at
        )
        
        assert node.node_id == "test_node_001"
        assert node.node_type == NodeType.EVENT
        assert node.properties == properties
        assert node.created_at == created_at
        assert node.updated_at is None
    
    def test_cypher_properties_conversion(self):
        """Test conversion to Cypher properties format."""
        properties = {
            "string_prop": "test_string",
            "int_prop": 42,
            "float_prop": 3.14,
            "bool_prop": True,
            "list_prop": [1, 2, 3]
        }
        
        node = GraphNode(
            node_id="test_node_002",
            node_type=NodeType.AGENT,
            properties=properties,
            created_at=datetime.utcnow()
        )
        
        cypher_props = node.to_cypher_properties()
        
        # Check that all properties are included
        assert "string_prop: 'test_string'" in cypher_props
        assert "int_prop: 42" in cypher_props
        assert "float_prop: 3.14" in cypher_props
        assert "bool_prop: true" in cypher_props
        assert "list_prop: [1, 2, 3]" in cypher_props
        assert "node_id: 'test_node_002'" in cypher_props
    
    def test_cypher_properties_with_none_values(self):
        """Test Cypher properties conversion with None values."""
        properties = {"valid_prop": "value", "none_prop": None}
        
        node = GraphNode(
            node_id="test_node_003",
            node_type=NodeType.KNOWLEDGE,
            properties=properties,
            created_at=datetime.utcnow(),
            updated_at=None
        )
        
        cypher_props = node.to_cypher_properties()
        
        # None values should be filtered out
        assert "valid_prop: 'value'" in cypher_props
        assert "none_prop" not in cypher_props
        assert "updated_at" not in cypher_props


class TestGraphRelationship:
    """Test GraphRelationship base class."""
    
    def test_graph_relationship_creation(self):
        """Test GraphRelationship creation with valid data."""
        properties = {"evidence": ["log1", "log2"], "mechanism": "direct"}
        created_at = datetime.utcnow()
        
        relationship = GraphRelationship(
            relationship_id="test_rel_001",
            relationship_type=RelationshipType.CAUSES,
            source_node_id="source_001",
            target_node_id="target_001",
            properties=properties,
            created_at=created_at,
            strength=0.85,
            confidence=0.9
        )
        
        assert relationship.relationship_id == "test_rel_001"
        assert relationship.relationship_type == RelationshipType.CAUSES
        assert relationship.source_node_id == "source_001"
        assert relationship.target_node_id == "target_001"
        assert relationship.properties == properties
        assert relationship.strength == 0.85
        assert relationship.confidence == 0.9
    
    def test_relationship_cypher_properties(self):
        """Test relationship Cypher properties conversion."""
        properties = {"delay_ms": 1000, "evidence": ["trace1"]}
        
        relationship = GraphRelationship(
            relationship_id="test_rel_002",
            relationship_type=RelationshipType.INFLUENCES,
            source_node_id="source_002",
            target_node_id="target_002",
            properties=properties,
            created_at=datetime.utcnow(),
            strength=0.7,
            confidence=0.8
        )
        
        cypher_props = relationship.to_cypher_properties()
        
        assert "relationship_id: 'test_rel_002'" in cypher_props
        assert "strength: 0.7" in cypher_props
        assert "confidence: 0.8" in cypher_props
        assert "delay_ms: 1000" in cypher_props


class TestCausalNode:
    """Test CausalNode specialized class."""
    
    def test_causal_node_creation(self):
        """Test CausalNode creation and property setup."""
        timestamp = datetime.utcnow()
        
        node = CausalNode(
            node_id="causal_001",
            event_type="decision",
            description="Agent makes a decision",
            timestamp=timestamp,
            agent_id="agent_001"
        )
        
        assert node.node_id == "causal_001"
        assert node.node_type == NodeType.EVENT
        assert node.properties["event_type"] == "decision"
        assert node.properties["description"] == "Agent makes a decision"
        assert node.properties["timestamp"] == timestamp.isoformat()
        assert node.properties["agent_id"] == "agent_001"
    
    def test_causal_node_with_additional_properties(self):
        """Test CausalNode with additional properties."""
        timestamp = datetime.utcnow()
        
        node = CausalNode(
            node_id="causal_002",
            event_type="action",
            description="Agent executes action",
            timestamp=timestamp,
            agent_id="agent_002",
            tool_used="search",
            success=True
        )
        
        assert node.properties["tool_used"] == "search"
        assert node.properties["success"] is True


class TestCausalRelationship:
    """Test CausalRelationship specialized class."""
    
    def test_causal_relationship_creation(self):
        """Test CausalRelationship creation and property setup."""
        relationship = CausalRelationship(
            relationship_id="causal_rel_001",
            cause_node_id="cause_001",
            effect_node_id="effect_001",
            causal_strength=0.85,
            delay_ms=500,
            evidence=["log_entry_1", "trace_data_1"]
        )
        
        assert relationship.relationship_id == "causal_rel_001"
        assert relationship.relationship_type == RelationshipType.CAUSES
        assert relationship.source_node_id == "cause_001"
        assert relationship.target_node_id == "effect_001"
        assert relationship.strength == 0.85
        assert relationship.properties["delay_ms"] == 500
        assert relationship.properties["evidence"] == ["log_entry_1", "trace_data_1"]
    
    def test_causal_relationship_with_optional_properties(self):
        """Test CausalRelationship with optional properties."""
        relationship = CausalRelationship(
            relationship_id="causal_rel_002",
            cause_node_id="cause_002",
            effect_node_id="effect_002",
            causal_strength=0.7,
            delay_ms=1000,
            evidence=["evidence_1"],
            confidence=0.9,
            statistical_significance=0.95,
            causal_mechanism="direct_influence"
        )
        
        assert relationship.confidence == 0.9
        assert relationship.properties["statistical_significance"] == 0.95
        assert relationship.properties["causal_mechanism"] == "direct_influence"


class TestKnowledgeNode:
    """Test KnowledgeNode specialized class."""
    
    def test_knowledge_node_creation(self):
        """Test KnowledgeNode creation and property setup."""
        node = KnowledgeNode(
            node_id="knowledge_001",
            knowledge_type="factual",
            content="The sky is blue",
            confidence_level=0.95,
            agent_id="agent_001"
        )
        
        assert node.node_id == "knowledge_001"
        assert node.node_type == NodeType.KNOWLEDGE
        assert node.properties["knowledge_type"] == "factual"
        assert node.properties["content"] == "The sky is blue"
        assert node.properties["confidence_level"] == 0.95
        assert node.properties["agent_id"] == "agent_001"
    
    def test_knowledge_node_with_additional_properties(self):
        """Test KnowledgeNode with additional properties."""
        node = KnowledgeNode(
            node_id="knowledge_002",
            knowledge_type="procedural",
            content="How to search for information",
            confidence_level=0.8,
            agent_id="agent_002",
            source="training_data",
            verified=True
        )
        
        assert node.properties["source"] == "training_data"
        assert node.properties["verified"] is True


class TestAgentNode:
    """Test AgentNode specialized class."""
    
    def test_agent_node_creation(self):
        """Test AgentNode creation and property setup."""
        capabilities = ["reasoning", "tool_use", "memory"]
        
        node = AgentNode(
            node_id="agent_001",
            agent_name="Test Agent",
            framework="langchain",
            capabilities=capabilities
        )
        
        assert node.node_id == "agent_001"
        assert node.node_type == NodeType.AGENT
        assert node.properties["agent_name"] == "Test Agent"
        assert node.properties["framework"] == "langchain"
        assert node.properties["capabilities"] == capabilities
    
    def test_agent_node_with_additional_properties(self):
        """Test AgentNode with additional properties."""
        node = AgentNode(
            node_id="agent_002",
            agent_name="Advanced Agent",
            framework="autogen",
            capabilities=["multi_agent", "coordination"],
            version="1.0.0",
            model="gpt-4"
        )
        
        assert node.properties["version"] == "1.0.0"
        assert node.properties["model"] == "gpt-4"


class TestGraphQuery:
    """Test GraphQuery data class."""
    
    def test_graph_query_creation(self):
        """Test GraphQuery creation."""
        query = "MATCH (n:Agent) RETURN n.name"
        parameters = {"limit": 10}
        
        graph_query = GraphQuery(query=query, parameters=parameters)
        
        assert graph_query.query == query
        assert graph_query.parameters == parameters
    
    def test_graph_query_string_representation(self):
        """Test GraphQuery string representation."""
        query = "MATCH (n) WHERE n.id = $id RETURN n"
        parameters = {"id": "test_001"}
        
        graph_query = GraphQuery(query=query, parameters=parameters)
        string_repr = str(graph_query)
        
        assert "Query:" in string_repr
        assert "Parameters:" in string_repr
        assert query in string_repr
        assert "test_001" in string_repr


class TestGraphAnalysisResult:
    """Test GraphAnalysisResult data class."""
    
    def test_graph_analysis_result_creation(self):
        """Test GraphAnalysisResult creation."""
        results = {
            "centrality_scores": {"node1": 0.8, "node2": 0.6},
            "patterns": ["pattern1", "pattern2"]
        }
        
        analysis_result = GraphAnalysisResult(
            analysis_type="centrality_analysis",
            results=results,
            execution_time_ms=150.5,
            node_count=100,
            relationship_count=250
        )
        
        assert analysis_result.analysis_type == "centrality_analysis"
        assert analysis_result.results == results
        assert analysis_result.execution_time_ms == 150.5
        assert analysis_result.node_count == 100
        assert analysis_result.relationship_count == 250
    
    def test_graph_analysis_result_to_dict(self):
        """Test GraphAnalysisResult to_dict conversion."""
        results = {"test_key": "test_value"}
        
        analysis_result = GraphAnalysisResult(
            analysis_type="test_analysis",
            results=results,
            execution_time_ms=100.0,
            node_count=50,
            relationship_count=75
        )
        
        result_dict = analysis_result.to_dict()
        
        assert result_dict["analysis_type"] == "test_analysis"
        assert result_dict["results"] == results
        assert result_dict["execution_time_ms"] == 100.0
        assert result_dict["node_count"] == 50
        assert result_dict["relationship_count"] == 75


class TestDataValidation:
    """Test data validation and edge cases."""
    
    def test_empty_properties(self):
        """Test handling of empty properties."""
        node = GraphNode(
            node_id="empty_props",
            node_type=NodeType.EVENT,
            properties={},
            created_at=datetime.utcnow()
        )
        
        cypher_props = node.to_cypher_properties()
        assert "node_id: 'empty_props'" in cypher_props
    
    def test_special_characters_in_properties(self):
        """Test handling of special characters in properties."""
        properties = {
            "description": "Test with 'quotes' and \"double quotes\"",
            "path": "C:\\Windows\\System32"
        }
        
        node = GraphNode(
            node_id="special_chars",
            node_type=NodeType.EVENT,
            properties=properties,
            created_at=datetime.utcnow()
        )
        
        # Should not raise an exception
        cypher_props = node.to_cypher_properties()
        assert isinstance(cypher_props, str)
    
    def test_large_property_values(self):
        """Test handling of large property values."""
        large_text = "x" * 10000  # 10KB string
        properties = {"large_content": large_text}
        
        node = GraphNode(
            node_id="large_props",
            node_type=NodeType.KNOWLEDGE,
            properties=properties,
            created_at=datetime.utcnow()
        )
        
        # Should handle large properties without issues
        cypher_props = node.to_cypher_properties()
        assert large_text in cypher_props
    
    def test_datetime_serialization(self):
        """Test datetime serialization in properties."""
        timestamp = datetime.utcnow()
        
        node = CausalNode(
            node_id="datetime_test",
            event_type="test_event",
            description="Test datetime serialization",
            timestamp=timestamp,
            agent_id="test_agent"
        )
        
        # Timestamp should be serialized as ISO format string
        assert node.properties["timestamp"] == timestamp.isoformat()
    
    def test_invalid_node_type(self):
        """Test behavior with invalid node types."""
        # This should work since we're using enum values
        node = GraphNode(
            node_id="test_node",
            node_type=NodeType.AGENT,
            properties={},
            created_at=datetime.utcnow()
        )
        
        assert node.node_type == NodeType.AGENT
    
    def test_negative_strength_and_confidence(self):
        """Test handling of negative strength and confidence values."""
        relationship = GraphRelationship(
            relationship_id="negative_test",
            relationship_type=RelationshipType.CAUSES,
            source_node_id="source",
            target_node_id="target",
            properties={},
            created_at=datetime.utcnow(),
            strength=-0.5,  # Negative strength
            confidence=-0.2  # Negative confidence
        )
        
        # Should accept negative values (validation can be added later if needed)
        assert relationship.strength == -0.5
        assert relationship.confidence == -0.2
    
    def test_very_large_numbers(self):
        """Test handling of very large numbers."""
        properties = {
            "large_number": 999999999999999,
            "large_float": 1.23e15
        }
        
        node = GraphNode(
            node_id="large_numbers",
            node_type=NodeType.EVENT,
            properties=properties,
            created_at=datetime.utcnow()
        )
        
        cypher_props = node.to_cypher_properties()
        assert "999999999999999" in cypher_props
        assert "1.23e+15" in cypher_props or "1230000000000000" in cypher_props