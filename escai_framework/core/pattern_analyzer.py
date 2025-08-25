"""
Behavioral pattern analyzer for ESCAI Framework.
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import asyncio

from ..models.behavioral_pattern import BehavioralPattern, ExecutionSequence
from ..analytics.pattern_mining import PatternMiner
from ..storage.repositories.behavioral_pattern_repository import BehavioralPatternRepository
from ..utils.logging import get_logger

logger = get_logger(__name__)

class BehavioralAnalyzer:
    """Analyzes behavioral patterns in agent execution."""
    
    def __init__(self):
        self.pattern_miner = PatternMiner()
        self.pattern_repository = BehavioralPatternRepository()
    
    async def analyze_patterns(
        self, 
        filters: Dict[str, Any], 
        page: int = 1, 
        size: int = 20
    ) -> Dict[str, Any]:
        """Analyze behavioral patterns with filtering and pagination."""
        try:
            # Get patterns from repository
            patterns = await self.pattern_repository.search_patterns(
                filters=filters,
                page=page,
                size=size
            )
            
            return {
                "items": [pattern.dict() for pattern in patterns["items"]],
                "total": patterns["total"]
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze patterns: {e}")
            raise
    
    async def detect_anomalies(
        self, 
        agent_id: str, 
        current_sequence: ExecutionSequence
    ) -> Dict[str, Any]:
        """Detect anomalies in current execution sequence."""
        try:
            # Get historical patterns for this agent
            historical_patterns = await self.pattern_repository.get_agent_patterns(agent_id)
            
            # Use pattern miner to detect anomalies
            anomaly_score = await self.pattern_miner.detect_anomaly(
                current_sequence, 
                historical_patterns
            )
            
            return {
                "agent_id": agent_id,
                "anomaly_score": anomaly_score,
                "is_anomalous": anomaly_score > 0.7,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to detect anomalies: {e}")
            raise
    
    async def mine_new_patterns(
        self, 
        agent_id: str, 
        sequences: List[ExecutionSequence]
    ) -> List[BehavioralPattern]:
        """Mine new behavioral patterns from execution sequences."""
        try:
            # Use pattern mining algorithm
            patterns = await self.pattern_miner.mine_patterns(sequences)
            
            # Store patterns in repository
            stored_patterns = []
            for pattern in patterns:
                pattern.agent_id = agent_id
                pattern.discovered_at = datetime.utcnow()
                stored_pattern = await self.pattern_repository.create_pattern(pattern)
                stored_patterns.append(stored_pattern)
            
            logger.info(f"Mined {len(stored_patterns)} new patterns for agent {agent_id}")
            return stored_patterns
            
        except Exception as e:
            logger.error(f"Failed to mine patterns: {e}")
            raise
    
    async def get_pattern_statistics(self, agent_id: str) -> Dict[str, Any]:
        """Get pattern statistics for an agent."""
        try:
            patterns = await self.pattern_repository.get_agent_patterns(agent_id)
            
            if not patterns:
                return {
                    "agent_id": agent_id,
                    "total_patterns": 0,
                    "average_success_rate": 0.0,
                    "most_common_pattern": None,
                    "pattern_diversity": 0.0
                }
            
            # Calculate statistics
            total_patterns = len(patterns)
            success_rates = [p.success_rate for p in patterns]
            average_success_rate = sum(success_rates) / len(success_rates)
            
            # Find most common pattern
            most_common = max(patterns, key=lambda p: p.frequency)
            
            # Calculate pattern diversity (entropy-based)
            frequencies = [p.frequency for p in patterns]
            total_frequency = sum(frequencies)
            probabilities = [f / total_frequency for f in frequencies]
            diversity = -sum(p * math.log2(p) for p in probabilities if p > 0)
            
            return {
                "agent_id": agent_id,
                "total_patterns": total_patterns,
                "average_success_rate": average_success_rate,
                "most_common_pattern": {
                    "pattern_id": most_common.pattern_id,
                    "pattern_name": most_common.pattern_name,
                    "frequency": most_common.frequency,
                    "success_rate": most_common.success_rate
                },
                "pattern_diversity": diversity
            }
            
        except Exception as e:
            logger.error(f"Failed to get pattern statistics: {e}")
            raise