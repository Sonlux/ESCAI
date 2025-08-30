"""
Neo4j database manager for ESCAI Framework.

This module provides the Neo4j database connection and operations for storing
and querying causal relationships, knowledge graphs, and agent interactions.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import time

from neo4j import GraphDatabase, AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable, TransientError

from .neo4j_models import (
    GraphNode, GraphRelationship, GraphQuery, GraphAnalysisResult,
    NodeType, RelationshipType, CausalNode, CausalRelationship,
    KnowledgeNode, AgentNode
)


logger = logging.getLogger(__name__)


class Neo4jManager:
    """Manages Neo4j database connections and operations."""
    
    def __init__(self, uri: str, username: str, password: str, 
                 database: str = "neo4j", max_connection_lifetime: int = 3600):
        """
        Initialize Neo4j manager.
        
        Args:
            uri: Neo4j database URI
            username: Database username
            password: Database password
            database: Database name
            max_connection_lifetime: Maximum connection lifetime in seconds
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver: Optional[Any] = None
        self.max_connection_lifetime = max_connection_lifetime
        
    async def connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_lifetime=self.max_connection_lifetime
            )
            # Test connection
            await self.driver.verify_connectivity()
            logger.info("Connected to Neo4j database")
            
            # Initialize schema
            await self._initialize_schema()
            
        except ServiceUnavailable as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error connecting to Neo4j: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close Neo4j database connection."""
        if self.driver:
            await self.driver.close()
            logger.info("Disconnected from Neo4j database")
    
    async def _initialize_schema(self) -> None:
        """Initialize Neo4j schema with constraints and indexes."""
        constraints_and_indexes = [
            # Unique constraints
            "CREATE CONSTRAINT agent_id_unique IF NOT EXISTS FOR (a:Agent) REQUIRE a.node_id IS UNIQUE",
            "CREATE CONSTRAINT event_id_unique IF NOT EXISTS FOR (e:Event) REQUIRE e.node_id IS UNIQUE",
            "CREATE CONSTRAINT knowledge_id_unique IF NOT EXISTS FOR (k:Knowledge) REQUIRE k.node_id IS UNIQUE",
            "CREATE CONSTRAINT belief_id_unique IF NOT EXISTS FOR (b:Belief) REQUIRE b.node_id IS UNIQUE",
            "CREATE CONSTRAINT goal_id_unique IF NOT EXISTS FOR (g:Goal) REQUIRE g.node_id IS UNIQUE",
            
            # Indexes for performance
            "CREATE INDEX agent_name_index IF NOT EXISTS FOR (a:Agent) ON (a.agent_name)",
            "CREATE INDEX event_timestamp_index IF NOT EXISTS FOR (e:Event) ON (e.timestamp)",
            "CREATE INDEX event_agent_index IF NOT EXISTS FOR (e:Event) ON (e.agent_id)",
            "CREATE INDEX knowledge_agent_index IF NOT EXISTS FOR (k:Knowledge) ON (k.agent_id)",
            "CREATE INDEX relationship_strength_index IF NOT EXISTS FOR ()-[r:CAUSES]-() ON (r.strength)",
            "CREATE INDEX relationship_confidence_index IF NOT EXISTS FOR ()-[r]-() ON (r.confidence)",
        ]
        
        async with self.driver.session(database=self.database) as session:
            for constraint in constraints_and_indexes:
                try:
                    await session.run(constraint)
                    logger.debug(f"Applied schema: {constraint}")
                except Exception as e:
                    logger.warning(f"Schema application failed (may already exist): {e}")
    
    async def create_node(self, node: GraphNode) -> bool:
        """
        Create a node in the graph database.
        
        Args:
            node: GraphNode to create
            
        Returns:
            bool: True if successful, False otherwise
        """
        query = f"""
        CREATE (n:{node.node_type.value} {node.to_cypher_properties()})
        RETURN n.node_id as node_id
        """
        
        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(query)
                record = await result.single()
                if record:
                    logger.debug(f"Created node: {record['node_id']}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to create node {node.node_id}: {e}")
            return False
    
    async def create_relationship(self, relationship: GraphRelationship) -> bool:
        """
        Create a relationship between nodes.
        
        Args:
            relationship: GraphRelationship to create
            
        Returns:
            bool: True if successful, False otherwise
        """
        query = f"""
        MATCH (a), (b)
        WHERE a.node_id = $source_id AND b.node_id = $target_id
        CREATE (a)-[r:{relationship.relationship_type.value} {relationship.to_cypher_properties()}]->(b)
        RETURN r.relationship_id as relationship_id
        """
        
        parameters = {
            'source_id': relationship.source_node_id,
            'target_id': relationship.target_node_id
        }
        
        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(query, parameters)
                record = await result.single()
                if record:
                    logger.debug(f"Created relationship: {record['relationship_id']}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to create relationship {relationship.relationship_id}: {e}")
            return False
    
    async def find_causal_paths(self, start_node_id: str, end_node_id: str, 
                               max_depth: int = 5) -> List[Dict[str, Any]]:
        """
        Find causal paths between two nodes.
        
        Args:
            start_node_id: Starting node ID
            end_node_id: Ending node ID
            max_depth: Maximum path depth
            
        Returns:
            List of causal paths with their properties
        """
        query = f"""
        MATCH path = (start)-[:CAUSES*1..{max_depth}]->(end)
        WHERE start.node_id = $start_id AND end.node_id = $end_id
        RETURN path,
               [rel in relationships(path) | rel.strength] as strengths,
               [rel in relationships(path) | rel.confidence] as confidences,
               length(path) as path_length
        ORDER BY path_length ASC
        LIMIT 10
        """
        
        parameters = {
            'start_id': start_node_id,
            'end_id': end_node_id
        }
        
        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(query, parameters)
                paths = []
                async for record in result:
                    path_data = {
                        'path_length': record['path_length'],
                        'strengths': record['strengths'],
                        'confidences': record['confidences'],
                        'average_strength': sum(record['strengths']) / len(record['strengths']),
                        'average_confidence': sum(record['confidences']) / len(record['confidences'])
                    }
                    paths.append(path_data)
                return paths
        except Exception as e:
            logger.error(f"Failed to find causal paths: {e}")
            return []
    
    async def get_node_centrality(self, node_type: Optional[NodeType] = None) -> Dict[str, float]:
        """
        Calculate centrality measures for nodes.
        
        Args:
            node_type: Optional node type filter
            
        Returns:
            Dictionary mapping node IDs to centrality scores
        """
        node_filter = f":{node_type.value}" if node_type else ""
        
        query = f"""
        MATCH (n{node_filter})
        OPTIONAL MATCH (n)-[r]-()
        WITH n, count(r) as degree
        RETURN n.node_id as node_id, degree
        ORDER BY degree DESC
        """
        
        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(query)
                centrality = {}
                async for record in result:
                    centrality[record['node_id']] = record['degree']
                return centrality
        except Exception as e:
            logger.error(f"Failed to calculate centrality: {e}")
            return {}
    
    async def find_strongly_connected_components(self) -> List[List[str]]:
        """
        Find strongly connected components in the causal graph.
        
        Returns:
            List of components, each containing node IDs
        """
        query = """
        CALL gds.scc.stream('causal-graph')
        YIELD nodeId, componentId
        RETURN componentId, collect(gds.util.asNode(nodeId).node_id) as nodes
        ORDER BY size(nodes) DESC
        """
        
        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(query)
                components = []
                async for record in result:
                    components.append(record['nodes'])
                return components
        except Exception as e:
            logger.warning(f"SCC analysis failed (GDS may not be available): {e}")
            # Fallback to simple connected components
            return await self._find_connected_components_fallback()
    
    async def _find_connected_components_fallback(self) -> List[List[str]]:
        """Fallback method for finding connected components without GDS."""
        query = """
        MATCH (n)
        OPTIONAL MATCH path = (n)-[:CAUSES*]-(m)
        WITH n, collect(DISTINCT m.node_id) + [n.node_id] as component
        RETURN DISTINCT component
        ORDER BY size(component) DESC
        """
        
        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(query)
                components = []
                async for record in result:
                    if record['component']:
                        components.append(record['component'])
                return components
        except Exception as e:
            logger.error(f"Fallback connected components failed: {e}")
            return []
    
    async def analyze_causal_patterns(self, agent_id: Optional[str] = None) -> GraphAnalysisResult:
        """
        Analyze causal patterns in the graph.
        
        Args:
            agent_id: Optional agent ID filter
            
        Returns:
            GraphAnalysisResult with pattern analysis
        """
        start_time = time.time()
        
        agent_filter = "WHERE n.agent_id = $agent_id" if agent_id else ""
        parameters = {'agent_id': agent_id} if agent_id else {}
        
        # Count nodes and relationships
        count_query = f"""
        MATCH (n:Event) {agent_filter}
        OPTIONAL MATCH (n)-[r:CAUSES]->()
        RETURN count(DISTINCT n) as node_count, count(r) as rel_count
        """
        
        # Find most influential events
        influence_query = f"""
        MATCH (n:Event) {agent_filter}
        OPTIONAL MATCH (n)-[r:CAUSES]->()
        WITH n, count(r) as outgoing_causes, avg(r.strength) as avg_strength
        RETURN n.node_id as event_id, n.event_type as event_type,
               outgoing_causes, avg_strength
        ORDER BY outgoing_causes DESC, avg_strength DESC
        LIMIT 10
        """
        
        # Find causal chains
        chains_query = f"""
        MATCH path = (start:Event)-[:CAUSES*2..5]->(end:Event)
        {agent_filter.replace('n.', 'start.')}
        WITH path, [rel in relationships(path) | rel.strength] as strengths
        RETURN length(path) as chain_length,
               avg([s in strengths | s]) as avg_strength,
               count(*) as frequency
        ORDER BY chain_length, avg_strength DESC
        """
        
        try:
            async with self.driver.session(database=self.database) as session:
                # Get counts
                count_result = await session.run(count_query, parameters)
                count_record = await count_result.single()
                node_count = count_record['node_count'] if count_record else 0
                rel_count = count_record['rel_count'] if count_record else 0
                
                # Get influential events
                influence_result = await session.run(influence_query, parameters)
                influential_events = []
                async for record in influence_result:
                    influential_events.append({
                        'event_id': record['event_id'],
                        'event_type': record['event_type'],
                        'outgoing_causes': record['outgoing_causes'],
                        'avg_strength': record['avg_strength']
                    })
                
                # Get causal chains
                chains_result = await session.run(chains_query, parameters)
                causal_chains = []
                async for record in chains_result:
                    causal_chains.append({
                        'chain_length': record['chain_length'],
                        'avg_strength': record['avg_strength'],
                        'frequency': record['frequency']
                    })
                
                execution_time = (time.time() - start_time) * 1000
                
                return GraphAnalysisResult(
                    analysis_type="causal_patterns",
                    results={
                        'influential_events': influential_events,
                        'causal_chains': causal_chains,
                        'agent_id': agent_id
                    },
                    execution_time_ms=execution_time,
                    node_count=node_count,
                    relationship_count=rel_count
                )
                
        except Exception as e:
            logger.error(f"Failed to analyze causal patterns: {e}")
            return GraphAnalysisResult(
                analysis_type="causal_patterns",
                results={'error': str(e)},
                execution_time_ms=(time.time() - start_time) * 1000,
                node_count=0,
                relationship_count=0
            )
    
    async def execute_custom_query(self, query: GraphQuery) -> List[Dict[str, Any]]:
        """
        Execute a custom Cypher query.
        
        Args:
            query: GraphQuery object with query and parameters
            
        Returns:
            List of result records
        """
        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(query.query, query.parameters)
                records = []
                async for record in result:
                    records.append(dict(record))
                return records
        except Exception as e:
            logger.error(f"Failed to execute custom query: {e}")
            return []
    
    async def delete_node(self, node_id: str) -> bool:
        """
        Delete a node and all its relationships.
        
        Args:
            node_id: ID of node to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        query = """
        MATCH (n {node_id: $node_id})
        DETACH DELETE n
        RETURN count(n) as deleted_count
        """
        
        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(query, {'node_id': node_id})
                record = await result.single()
                deleted_count = record['deleted_count'] if record else 0
                return deleted_count > 0
        except Exception as e:
            logger.error(f"Failed to delete node {node_id}: {e}")
            return False
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get overall graph statistics.
        
        Returns:
            Dictionary with graph statistics
        """
        query = """
        MATCH (n)
        OPTIONAL MATCH ()-[r]->()
        RETURN 
            count(DISTINCT n) as total_nodes,
            count(r) as total_relationships,
            count(DISTINCT labels(n)) as node_types,
            count(DISTINCT type(r)) as relationship_types
        """
        
        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(query)
                record = await result.single()
                if record:
                    return {
                        'total_nodes': record['total_nodes'],
                        'total_relationships': record['total_relationships'],
                        'node_types': record['node_types'],
                        'relationship_types': record['relationship_types']
                    }
                return {}
        except Exception as e:
            logger.error(f"Failed to get graph statistics: {e}")
            return {}


# Convenience functions for common operations
async def create_causal_relationship_graph(manager: Neo4jManager, 
                                         cause_event: Dict[str, Any],
                                         effect_event: Dict[str, Any],
                                         relationship_data: Dict[str, Any]) -> bool:
    """
    Create a causal relationship between two events.
    
    Args:
        manager: Neo4jManager instance
        cause_event: Cause event data
        effect_event: Effect event data
        relationship_data: Relationship properties
        
    Returns:
        bool: True if successful
    """
    # Create cause node
    cause_node = CausalNode(
        node_id=cause_event['node_id'],
        event_type=cause_event['event_type'],
        description=cause_event['description'],
        timestamp=cause_event['timestamp'],
        agent_id=cause_event['agent_id']
    )
    
    # Create effect node
    effect_node = CausalNode(
        node_id=effect_event['node_id'],
        event_type=effect_event['event_type'],
        description=effect_event['description'],
        timestamp=effect_event['timestamp'],
        agent_id=effect_event['agent_id']
    )
    
    # Create relationship
    relationship = CausalRelationship(
        relationship_id=relationship_data['relationship_id'],
        cause_node_id=cause_event['node_id'],
        effect_node_id=effect_event['node_id'],
        causal_strength=relationship_data['strength'],
        delay_ms=relationship_data['delay_ms'],
        evidence=relationship_data['evidence'],
        confidence=relationship_data.get('confidence', 1.0)
    )
    
    # Execute operations
    success = True
    success &= await manager.create_node(cause_node)
    success &= await manager.create_node(effect_node)
    success &= await manager.create_relationship(relationship)
    
    return success