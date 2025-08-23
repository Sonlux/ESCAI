"""
Neo4j graph analytics and algorithms for ESCAI Framework.

This module provides advanced graph analysis algorithms including centrality measures,
pattern discovery, and graph visualization utilities for causal relationships.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from enum import Enum

import networkx as nx
import numpy as np
from collections import defaultdict, Counter

from .neo4j_manager import Neo4jManager
from .neo4j_models import GraphAnalysisResult, NodeType, RelationshipType


logger = logging.getLogger(__name__)


class CentralityMeasure(Enum):
    """Types of centrality measures."""
    DEGREE = "degree"
    BETWEENNESS = "betweenness"
    CLOSENESS = "closeness"
    EIGENVECTOR = "eigenvector"
    PAGERANK = "pagerank"


@dataclass
class GraphPattern:
    """Represents a discovered graph pattern."""
    pattern_id: str
    pattern_type: str
    nodes: List[str]
    relationships: List[str]
    frequency: int
    significance_score: float
    description: str


@dataclass
class CausalChain:
    """Represents a causal chain in the graph."""
    chain_id: str
    nodes: List[str]
    relationships: List[str]
    total_strength: float
    average_confidence: float
    chain_length: int
    start_event: str
    end_event: str


class Neo4jAnalytics:
    """Advanced analytics for Neo4j graph data."""
    
    def __init__(self, manager: Neo4jManager):
        """
        Initialize analytics with Neo4j manager.
        
        Args:
            manager: Neo4jManager instance
        """
        self.manager = manager
        
    async def calculate_centrality_measures(self, 
                                          measure: CentralityMeasure,
                                          node_type: Optional[NodeType] = None,
                                          limit: int = 50) -> Dict[str, float]:
        """
        Calculate centrality measures for nodes.
        
        Args:
            measure: Type of centrality measure
            node_type: Optional node type filter
            limit: Maximum number of results
            
        Returns:
            Dictionary mapping node IDs to centrality scores
        """
        if measure == CentralityMeasure.DEGREE:
            return await self._calculate_degree_centrality(node_type, limit)
        elif measure == CentralityMeasure.BETWEENNESS:
            return await self._calculate_betweenness_centrality(node_type, limit)
        elif measure == CentralityMeasure.CLOSENESS:
            return await self._calculate_closeness_centrality(node_type, limit)
        elif measure == CentralityMeasure.PAGERANK:
            return await self._calculate_pagerank_centrality(node_type, limit)
        else:
            logger.warning(f"Centrality measure {measure} not implemented")
            return {}
    
    async def _calculate_degree_centrality(self, 
                                         node_type: Optional[NodeType],
                                         limit: int) -> Dict[str, float]:
        """Calculate degree centrality."""
        node_filter = f":{node_type.value}" if node_type else ""
        
        query = f"""
        MATCH (n{node_filter})
        OPTIONAL MATCH (n)-[r]-(m)
        WITH n, count(DISTINCT r) as degree
        RETURN n.node_id as node_id, degree
        ORDER BY degree DESC
        LIMIT {limit}
        """
        
        try:
            results = await self.manager.execute_custom_query(
                type('GraphQuery', (), {'query': query, 'parameters': {}})()
            )
            
            # Normalize scores
            max_degree = max([r['degree'] for r in results]) if results else 1
            return {
                r['node_id']: r['degree'] / max_degree 
                for r in results
            }
        except Exception as e:
            logger.error(f"Failed to calculate degree centrality: {e}")
            return {}
    
    async def _calculate_betweenness_centrality(self, 
                                              node_type: Optional[NodeType],
                                              limit: int) -> Dict[str, float]:
        """Calculate betweenness centrality using NetworkX."""
        # Get graph data
        graph_data = await self._get_graph_as_networkx(node_type)
        if not graph_data:
            return {}
        
        try:
            # Calculate betweenness centrality
            centrality = nx.betweenness_centrality(graph_data)
            
            # Sort and limit results
            sorted_centrality = sorted(
                centrality.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:limit]
            
            return dict(sorted_centrality)
        except Exception as e:
            logger.error(f"Failed to calculate betweenness centrality: {e}")
            return {}
    
    async def _calculate_closeness_centrality(self, 
                                            node_type: Optional[NodeType],
                                            limit: int) -> Dict[str, float]:
        """Calculate closeness centrality using NetworkX."""
        graph_data = await self._get_graph_as_networkx(node_type)
        if not graph_data:
            return {}
        
        try:
            centrality = nx.closeness_centrality(graph_data)
            sorted_centrality = sorted(
                centrality.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:limit]
            
            return dict(sorted_centrality)
        except Exception as e:
            logger.error(f"Failed to calculate closeness centrality: {e}")
            return {}
    
    async def _calculate_pagerank_centrality(self, 
                                           node_type: Optional[NodeType],
                                           limit: int) -> Dict[str, float]:
        """Calculate PageRank centrality using NetworkX."""
        graph_data = await self._get_graph_as_networkx(node_type)
        if not graph_data:
            return {}
        
        try:
            centrality = nx.pagerank(graph_data)
            sorted_centrality = sorted(
                centrality.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:limit]
            
            return dict(sorted_centrality)
        except Exception as e:
            logger.error(f"Failed to calculate PageRank centrality: {e}")
            return {}
    
    async def _get_graph_as_networkx(self, 
                                   node_type: Optional[NodeType] = None) -> nx.DiGraph:
        """Convert Neo4j graph to NetworkX for analysis."""
        node_filter = f":{node_type.value}" if node_type else ""
        
        query = f"""
        MATCH (n{node_filter})-[r]->(m{node_filter})
        RETURN n.node_id as source, m.node_id as target, 
               r.strength as weight, type(r) as rel_type
        """
        
        try:
            results = await self.manager.execute_custom_query(
                type('GraphQuery', (), {'query': query, 'parameters': {}})()
            )
            
            # Create NetworkX graph
            G = nx.DiGraph()
            for result in results:
                G.add_edge(
                    result['source'], 
                    result['target'],
                    weight=result.get('weight', 1.0),
                    rel_type=result.get('rel_type', 'UNKNOWN')
                )
            
            return G
        except Exception as e:
            logger.error(f"Failed to convert graph to NetworkX: {e}")
            return nx.DiGraph()
    
    async def discover_causal_patterns(self, 
                                     agent_id: Optional[str] = None,
                                     min_frequency: int = 3,
                                     min_significance: float = 0.7) -> List[GraphPattern]:
        """
        Discover recurring causal patterns in the graph.
        
        Args:
            agent_id: Optional agent ID filter
            min_frequency: Minimum pattern frequency
            min_significance: Minimum significance score
            
        Returns:
            List of discovered patterns
        """
        agent_filter = "AND n.agent_id = $agent_id" if agent_id else ""
        parameters = {'agent_id': agent_id} if agent_id else {}
        
        # Find common causal sequences
        query = f"""
        MATCH path = (n1:Event)-[:CAUSES]->(n2:Event)-[:CAUSES]->(n3:Event)
        WHERE 1=1 {agent_filter}
        WITH [n1.event_type, n2.event_type, n3.event_type] as pattern,
             count(*) as frequency,
             avg([rel in relationships(path) | rel.strength]) as avg_strength
        WHERE frequency >= $min_frequency
        RETURN pattern, frequency, avg_strength
        ORDER BY frequency DESC, avg_strength DESC
        LIMIT 20
        """
        
        parameters.update({'min_frequency': min_frequency})
        
        try:
            results = await self.manager.execute_custom_query(
                type('GraphQuery', (), {'query': query, 'parameters': parameters})()
            )
            
            patterns = []
            for i, result in enumerate(results):
                if result['avg_strength'] >= min_significance:
                    pattern = GraphPattern(
                        pattern_id=f"pattern_{i}",
                        pattern_type="causal_sequence",
                        nodes=result['pattern'],
                        relationships=["CAUSES", "CAUSES"],
                        frequency=result['frequency'],
                        significance_score=result['avg_strength'],
                        description=f"Causal sequence: {' -> '.join(result['pattern'])}"
                    )
                    patterns.append(pattern)
            
            return patterns
        except Exception as e:
            logger.error(f"Failed to discover causal patterns: {e}")
            return []
    
    async def find_causal_chains(self, 
                               start_event_type: Optional[str] = None,
                               end_event_type: Optional[str] = None,
                               max_length: int = 5,
                               min_strength: float = 0.5) -> List[CausalChain]:
        """
        Find causal chains in the graph.
        
        Args:
            start_event_type: Optional starting event type filter
            end_event_type: Optional ending event type filter
            max_length: Maximum chain length
            min_strength: Minimum average strength
            
        Returns:
            List of causal chains
        """
        start_filter = "AND start.event_type = $start_type" if start_event_type else ""
        end_filter = "AND end.event_type = $end_type" if end_event_type else ""
        
        parameters = {}
        if start_event_type:
            parameters['start_type'] = start_event_type
        if end_event_type:
            parameters['end_type'] = end_event_type
        
        query = f"""
        MATCH path = (start:Event)-[:CAUSES*1..{max_length}]->(end:Event)
        WHERE start <> end {start_filter} {end_filter}
        WITH path, 
             [rel in relationships(path) | rel.strength] as strengths,
             [rel in relationships(path) | rel.confidence] as confidences,
             [node in nodes(path) | node.node_id] as node_ids,
             [rel in relationships(path) | rel.relationship_id] as rel_ids
        WHERE all(s in strengths WHERE s >= $min_strength)
        RETURN node_ids, rel_ids, strengths, confidences,
               reduce(total = 0, s in strengths | total + s) as total_strength,
               reduce(total = 0, c in confidences | total + c) / size(confidences) as avg_confidence,
               length(path) as chain_length,
               head(node_ids) as start_event,
               last(node_ids) as end_event
        ORDER BY total_strength DESC, avg_confidence DESC
        LIMIT 50
        """
        
        parameters['min_strength'] = min_strength
        
        try:
            results = await self.manager.execute_custom_query(
                type('GraphQuery', (), {'query': query, 'parameters': parameters})()
            )
            
            chains = []
            for i, result in enumerate(results):
                chain = CausalChain(
                    chain_id=f"chain_{i}",
                    nodes=result['node_ids'],
                    relationships=result['rel_ids'],
                    total_strength=result['total_strength'],
                    average_confidence=result['avg_confidence'],
                    chain_length=result['chain_length'],
                    start_event=result['start_event'],
                    end_event=result['end_event']
                )
                chains.append(chain)
            
            return chains
        except Exception as e:
            logger.error(f"Failed to find causal chains: {e}")
            return []
    
    async def analyze_temporal_patterns(self, 
                                      time_window_hours: int = 24,
                                      agent_id: Optional[str] = None) -> GraphAnalysisResult:
        """
        Analyze temporal patterns in causal relationships.
        
        Args:
            time_window_hours: Time window for analysis
            agent_id: Optional agent ID filter
            
        Returns:
            GraphAnalysisResult with temporal analysis
        """
        start_time = datetime.utcnow()
        
        agent_filter = "AND n1.agent_id = $agent_id" if agent_id else ""
        parameters = {'hours': time_window_hours}
        if agent_id:
            parameters['agent_id'] = agent_id
        
        # Analyze causal relationships by time of day
        query = f"""
        MATCH (n1:Event)-[r:CAUSES]->(n2:Event)
        WHERE datetime(n1.timestamp) > datetime() - duration({{hours: $hours}})
        {agent_filter}
        WITH r, n1, n2,
             datetime(n1.timestamp).hour as hour_of_day,
             r.delay_ms as delay
        RETURN hour_of_day,
               count(*) as causal_events,
               avg(r.strength) as avg_strength,
               avg(delay) as avg_delay_ms,
               collect(DISTINCT n1.event_type) as cause_types,
               collect(DISTINCT n2.event_type) as effect_types
        ORDER BY hour_of_day
        """
        
        try:
            results = await self.manager.execute_custom_query(
                type('GraphQuery', (), {'query': query, 'parameters': parameters})()
            )
            
            temporal_patterns = []
            for result in results:
                temporal_patterns.append({
                    'hour_of_day': result['hour_of_day'],
                    'causal_events': result['causal_events'],
                    'avg_strength': result['avg_strength'],
                    'avg_delay_ms': result['avg_delay_ms'],
                    'cause_types': result['cause_types'],
                    'effect_types': result['effect_types']
                })
            
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return GraphAnalysisResult(
                analysis_type="temporal_patterns",
                results={
                    'temporal_patterns': temporal_patterns,
                    'time_window_hours': time_window_hours,
                    'agent_id': agent_id
                },
                execution_time_ms=execution_time,
                node_count=len(temporal_patterns),
                relationship_count=sum(p['causal_events'] for p in temporal_patterns)
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze temporal patterns: {e}")
            return GraphAnalysisResult(
                analysis_type="temporal_patterns",
                results={'error': str(e)},
                execution_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                node_count=0,
                relationship_count=0
            )
    
    async def detect_anomalous_patterns(self, 
                                      agent_id: Optional[str] = None,
                                      threshold_std: float = 2.0) -> List[Dict[str, Any]]:
        """
        Detect anomalous causal patterns using statistical analysis.
        
        Args:
            agent_id: Optional agent ID filter
            threshold_std: Standard deviation threshold for anomaly detection
            
        Returns:
            List of anomalous patterns
        """
        agent_filter = "AND n1.agent_id = $agent_id" if agent_id else ""
        parameters = {'agent_id': agent_id} if agent_id else {}
        
        # Get causal relationship statistics
        query = f"""
        MATCH (n1:Event)-[r:CAUSES]->(n2:Event)
        WHERE 1=1 {agent_filter}
        WITH n1.event_type as cause_type, n2.event_type as effect_type,
             collect(r.strength) as strengths,
             collect(r.delay_ms) as delays,
             count(*) as frequency
        WHERE frequency >= 3
        RETURN cause_type, effect_type, frequency,
               reduce(sum = 0.0, s in strengths | sum + s) / size(strengths) as avg_strength,
               reduce(sum = 0.0, d in delays | sum + d) / size(delays) as avg_delay,
               strengths, delays
        """
        
        try:
            results = await self.manager.execute_custom_query(
                type('GraphQuery', (), {'query': query, 'parameters': parameters})()
            )
            
            # Calculate statistics for anomaly detection
            all_strengths = []
            all_delays = []
            for result in results:
                all_strengths.extend(result['strengths'])
                all_delays.extend(result['delays'])
            
            if not all_strengths:
                return []
            
            strength_mean = np.mean(all_strengths)
            strength_std = np.std(all_strengths)
            delay_mean = np.mean(all_delays)
            delay_std = np.std(all_delays)
            
            anomalies = []
            for result in results:
                # Check for strength anomalies
                strength_z_score = abs(result['avg_strength'] - strength_mean) / strength_std
                delay_z_score = abs(result['avg_delay'] - delay_mean) / delay_std
                
                if strength_z_score > threshold_std or delay_z_score > threshold_std:
                    anomaly = {
                        'cause_type': result['cause_type'],
                        'effect_type': result['effect_type'],
                        'frequency': result['frequency'],
                        'avg_strength': result['avg_strength'],
                        'avg_delay': result['avg_delay'],
                        'strength_z_score': strength_z_score,
                        'delay_z_score': delay_z_score,
                        'anomaly_type': 'strength' if strength_z_score > threshold_std else 'delay'
                    }
                    anomalies.append(anomaly)
            
            return sorted(anomalies, key=lambda x: max(x['strength_z_score'], x['delay_z_score']), reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to detect anomalous patterns: {e}")
            return []
    
    async def generate_graph_visualization_data(self, 
                                              agent_id: Optional[str] = None,
                                              max_nodes: int = 100) -> Dict[str, Any]:
        """
        Generate data for graph visualization.
        
        Args:
            agent_id: Optional agent ID filter
            max_nodes: Maximum number of nodes to include
            
        Returns:
            Dictionary with nodes and edges for visualization
        """
        agent_filter = "AND n.agent_id = $agent_id" if agent_id else ""
        parameters = {'limit': max_nodes}
        if agent_id:
            parameters['agent_id'] = agent_id
        
        # Get nodes with their properties
        nodes_query = f"""
        MATCH (n:Event)
        WHERE 1=1 {agent_filter}
        OPTIONAL MATCH (n)-[r]-(m)
        WITH n, count(r) as degree
        RETURN n.node_id as id, n.event_type as type, n.description as description,
               n.timestamp as timestamp, n.agent_id as agent_id, degree
        ORDER BY degree DESC
        LIMIT $limit
        """
        
        # Get relationships between selected nodes
        relationships_query = f"""
        MATCH (n1:Event)-[r:CAUSES]->(n2:Event)
        WHERE n1.node_id IN $node_ids AND n2.node_id IN $node_ids
        RETURN n1.node_id as source, n2.node_id as target,
               r.strength as strength, r.confidence as confidence,
               r.delay_ms as delay, type(r) as type
        """
        
        try:
            # Get nodes
            nodes_results = await self.manager.execute_custom_query(
                type('GraphQuery', (), {'query': nodes_query, 'parameters': parameters})()
            )
            
            if not nodes_results:
                return {'nodes': [], 'edges': []}
            
            # Extract node IDs for relationship query
            node_ids = [node['id'] for node in nodes_results]
            
            # Get relationships
            rel_parameters = {'node_ids': node_ids}
            relationships_results = await self.manager.execute_custom_query(
                type('GraphQuery', (), {'query': relationships_query, 'parameters': rel_parameters})()
            )
            
            # Format for visualization
            nodes = []
            for node in nodes_results:
                nodes.append({
                    'id': node['id'],
                    'label': f"{node['type']}\n{node['id'][:8]}",
                    'type': node['type'],
                    'description': node['description'],
                    'timestamp': node['timestamp'],
                    'agent_id': node['agent_id'],
                    'size': min(10 + node['degree'] * 2, 50),  # Scale node size by degree
                    'color': self._get_node_color(node['type'])
                })
            
            edges = []
            for rel in relationships_results:
                edges.append({
                    'source': rel['source'],
                    'target': rel['target'],
                    'strength': rel['strength'],
                    'confidence': rel['confidence'],
                    'delay': rel['delay'],
                    'type': rel['type'],
                    'width': max(1, rel['strength'] * 5),  # Scale edge width by strength
                    'color': self._get_edge_color(rel['strength'])
                })
            
            return {
                'nodes': nodes,
                'edges': edges,
                'metadata': {
                    'total_nodes': len(nodes),
                    'total_edges': len(edges),
                    'agent_id': agent_id
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to generate visualization data: {e}")
            return {'nodes': [], 'edges': [], 'error': str(e)}
    
    def _get_node_color(self, node_type: str) -> str:
        """Get color for node type."""
        color_map = {
            'decision': '#FF6B6B',
            'action': '#4ECDC4',
            'outcome': '#45B7D1',
            'belief': '#96CEB4',
            'goal': '#FFEAA7',
            'knowledge': '#DDA0DD',
            'error': '#FF7675',
            'success': '#00B894'
        }
        return color_map.get(node_type.lower(), '#74B9FF')
    
    def _get_edge_color(self, strength: float) -> str:
        """Get color for edge based on strength."""
        if strength >= 0.8:
            return '#E17055'  # Strong - red
        elif strength >= 0.6:
            return '#FDCB6E'  # Medium - yellow
        else:
            return '#74B9FF'  # Weak - blue