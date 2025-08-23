"""
Neo4j graph models for ESCAI Framework.

This module defines the graph data models for storing causal relationships,
knowledge graphs, and agent interaction networks in Neo4j.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class NodeType(Enum):
    """Types of nodes in the graph database."""
    AGENT = "Agent"
    EPISTEMIC_STATE = "EpistemicState"
    BELIEF = "Belief"
    KNOWLEDGE = "Knowledge"
    GOAL = "Goal"
    EVENT = "Event"
    DECISION = "Decision"
    ACTION = "Action"
    OUTCOME = "Outcome"
    PATTERN = "Pattern"


class RelationshipType(Enum):
    """Types of relationships in the graph database."""
    CAUSES = "CAUSES"
    INFLUENCES = "INFLUENCES"
    LEADS_TO = "LEADS_TO"
    DEPENDS_ON = "DEPENDS_ON"
    CONTAINS = "CONTAINS"
    FOLLOWS = "FOLLOWS"
    CORRELATES_WITH = "CORRELATES_WITH"
    CONTRADICTS = "CONTRADICTS"
    SUPPORTS = "SUPPORTS"
    TRIGGERS = "TRIGGERS"


@dataclass
class GraphNode:
    """Base class for graph nodes."""
    node_id: str
    node_type: NodeType
    properties: Dict[str, Any]
    created_at: datetime
    updated_at: Optional[datetime] = None

    def to_cypher_properties(self) -> str:
        """Convert properties to Cypher format."""
        props = {
            'node_id': self.node_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            **self.properties
        }
        # Filter out None values
        props = {k: v for k, v in props.items() if v is not None}
        
        # Format for Cypher
        prop_strings = []
        for key, value in props.items():
            if isinstance(value, bool):  # Check bool first since bool is subclass of int
                prop_strings.append(f"{key}: {'true' if value else 'false'}")
            elif isinstance(value, str):
                prop_strings.append(f"{key}: '{value}'")
            elif isinstance(value, (int, float)):
                prop_strings.append(f"{key}: {value}")
            elif isinstance(value, list):
                prop_strings.append(f"{key}: {value}")
        
        return "{" + ", ".join(prop_strings) + "}"


@dataclass
class GraphRelationship:
    """Base class for graph relationships."""
    relationship_id: str
    relationship_type: RelationshipType
    source_node_id: str
    target_node_id: str
    properties: Dict[str, Any]
    created_at: datetime
    strength: float = 1.0
    confidence: float = 1.0

    def to_cypher_properties(self) -> str:
        """Convert properties to Cypher format."""
        props = {
            'relationship_id': self.relationship_id,
            'strength': self.strength,
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat(),
            **self.properties
        }
        
        # Format for Cypher
        prop_strings = []
        for key, value in props.items():
            if isinstance(value, bool):  # Check bool first since bool is subclass of int
                prop_strings.append(f"{key}: {'true' if value else 'false'}")
            elif isinstance(value, str):
                prop_strings.append(f"{key}: '{value}'")
            elif isinstance(value, (int, float)):
                prop_strings.append(f"{key}: {value}")
            elif isinstance(value, list):
                prop_strings.append(f"{key}: {value}")
        
        return "{" + ", ".join(prop_strings) + "}"


@dataclass
class CausalNode(GraphNode):
    """Node representing a causal event or state."""
    
    def __init__(self, node_id: str, event_type: str, description: str, 
                 timestamp: datetime, agent_id: str, **kwargs):
        properties = {
            'event_type': event_type,
            'description': description,
            'timestamp': timestamp.isoformat(),
            'agent_id': agent_id,
            **kwargs
        }
        super().__init__(
            node_id=node_id,
            node_type=NodeType.EVENT,
            properties=properties,
            created_at=datetime.utcnow()
        )


@dataclass
class CausalRelationship(GraphRelationship):
    """Relationship representing causal connections."""
    
    def __init__(self, relationship_id: str, cause_node_id: str, 
                 effect_node_id: str, causal_strength: float, 
                 delay_ms: int, evidence: List[str], **kwargs):
        properties = {
            'delay_ms': delay_ms,
            'evidence': evidence,
            'statistical_significance': kwargs.get('statistical_significance', 0.0),
            'causal_mechanism': kwargs.get('causal_mechanism', ''),
            **kwargs
        }
        super().__init__(
            relationship_id=relationship_id,
            relationship_type=RelationshipType.CAUSES,
            source_node_id=cause_node_id,
            target_node_id=effect_node_id,
            properties=properties,
            created_at=datetime.utcnow(),
            strength=causal_strength,
            confidence=kwargs.get('confidence', 1.0)
        )


@dataclass
class KnowledgeNode(GraphNode):
    """Node representing knowledge or belief states."""
    
    def __init__(self, node_id: str, knowledge_type: str, content: str,
                 confidence_level: float, agent_id: str, **kwargs):
        properties = {
            'knowledge_type': knowledge_type,
            'content': content,
            'confidence_level': confidence_level,
            'agent_id': agent_id,
            **kwargs
        }
        super().__init__(
            node_id=node_id,
            node_type=NodeType.KNOWLEDGE,
            properties=properties,
            created_at=datetime.utcnow()
        )


@dataclass
class AgentNode(GraphNode):
    """Node representing an agent."""
    
    def __init__(self, node_id: str, agent_name: str, framework: str,
                 capabilities: List[str], **kwargs):
        properties = {
            'agent_name': agent_name,
            'framework': framework,
            'capabilities': capabilities,
            **kwargs
        }
        super().__init__(
            node_id=node_id,
            node_type=NodeType.AGENT,
            properties=properties,
            created_at=datetime.utcnow()
        )


@dataclass
class GraphQuery:
    """Represents a Cypher query with parameters."""
    query: str
    parameters: Dict[str, Any]
    
    def __str__(self) -> str:
        return f"Query: {self.query}\nParameters: {self.parameters}"


@dataclass
class GraphAnalysisResult:
    """Result of graph analysis operations."""
    analysis_type: str
    results: Dict[str, Any]
    execution_time_ms: float
    node_count: int
    relationship_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'analysis_type': self.analysis_type,
            'results': self.results,
            'execution_time_ms': self.execution_time_ms,
            'node_count': self.node_count,
            'relationship_count': self.relationship_count
        }