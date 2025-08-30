"""
Pattern mining algorithms for behavioral sequence analysis.

This module implements PrefixSpan and SPADE algorithms for discovering
frequent sequential patterns in agent behavioral data.
"""

import asyncio
import logging
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, Counter
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

from ..models.behavioral_pattern import BehavioralPattern, ExecutionSequence


@dataclass
class SequentialPattern:
    """Represents a discovered sequential pattern."""
    pattern: List[str]
    support: int
    confidence: float
    frequency: int
    avg_duration: float


@dataclass
class PatternMiningConfig:
    """Configuration for pattern mining algorithms."""
    min_support: int = 2
    min_confidence: float = 0.5
    max_pattern_length: int = 10
    window_size: int = 100


class PrefixSpanMiner:
    """
    Implementation of PrefixSpan algorithm for sequential pattern mining.
    
    PrefixSpan finds frequent sequential patterns by recursively growing
    patterns from frequent prefixes.
    """
    
    def __init__(self, config: PatternMiningConfig):
        self.config = config
        self.frequent_patterns: List[SequentialPattern] = []
    
    async def mine_patterns(self, sequences: List[ExecutionSequence]) -> List[SequentialPattern]:
        """
        Mine frequent sequential patterns from execution sequences.
        
        Args:
            sequences: List of agent execution sequences
            
        Returns:
            List of discovered sequential patterns
        """
        # Convert sequences to item sequences
        item_sequences = self._convert_to_item_sequences(sequences)
        
        # Find frequent 1-patterns
        frequent_items = self._find_frequent_items(item_sequences)
        
        # Recursively mine patterns
        self.frequent_patterns = []
        for item in frequent_items:
            await self._mine_recursive([item], item_sequences)
        
        return self.frequent_patterns
    
    def _convert_to_item_sequences(self, sequences: List[ExecutionSequence]) -> List[List[str]]:
        """Convert execution sequences to item sequences."""
        item_sequences = []
        for seq in sequences:
            items = []
            for step in seq.steps:
                # Use step type and action as items
                items.append(f"{step.step_type}:{step.action}")
            item_sequences.append(items)
        return item_sequences
    
    def _find_frequent_items(self, sequences: List[List[str]]) -> List[str]:
        """Find items that meet minimum support threshold."""
        item_counts: Counter[str] = Counter()
        for sequence in sequences:
            unique_items = set(sequence)
            for item in unique_items:
                item_counts[item] += 1
        
        return [item for item, count in item_counts.items() 
                if count >= self.config.min_support]
    
    async def _mine_recursive(self, prefix: List[str], sequences: List[List[str]]):
        """Recursively mine patterns with given prefix."""
        if len(prefix) >= self.config.max_pattern_length:
            return
        
        # Find projected database
        projected_db = self._project_database(prefix, sequences)
        
        if len(projected_db) < self.config.min_support:
            return
        
        # Add current pattern if it meets criteria
        pattern = SequentialPattern(
            pattern=prefix.copy(),
            support=len(projected_db),
            confidence=len(projected_db) / len(sequences),
            frequency=len(projected_db),
            avg_duration=0.0  # Will be calculated later
        )
        self.frequent_patterns.append(pattern)
        
        # Find frequent items in projected database
        frequent_items = self._find_frequent_items(projected_db)
        
        # Recursively mine with extended prefixes
        for item in frequent_items:
            new_prefix = prefix + [item]
            await self._mine_recursive(new_prefix, projected_db)
    
    def _project_database(self, prefix: List[str], sequences: List[List[str]]) -> List[List[str]]:
        """Project database for given prefix."""
        projected = []
        
        for sequence in sequences:
            suffix = self._find_suffix(prefix, sequence)
            if suffix:
                projected.append(suffix)
        
        return projected
    
    def _find_suffix(self, prefix: List[str], sequence: List[str]) -> Optional[List[str]]:
        """Find suffix of sequence after matching prefix."""
        if not prefix:
            return sequence
        
        # Find first occurrence of prefix
        for i in range(len(sequence) - len(prefix) + 1):
            if sequence[i:i+len(prefix)] == prefix:
                return sequence[i+len(prefix):]
        
        return None


class SPADEMiner:
    """
    Implementation of SPADE (Sequential Pattern Discovery using Equivalence classes) algorithm.
    
    SPADE uses a vertical database format and equivalence classes to efficiently
    discover sequential patterns.
    """
    
    def __init__(self, config: PatternMiningConfig):
        self.config = config
        self.vertical_db: Dict[str, List[Tuple[int, int]]] = {}
        self.frequent_patterns: List[SequentialPattern] = []
    
    async def mine_patterns(self, sequences: List[ExecutionSequence]) -> List[SequentialPattern]:
        """
        Mine frequent sequential patterns using SPADE algorithm.
        
        Args:
            sequences: List of agent execution sequences
            
        Returns:
            List of discovered sequential patterns
        """
        # Build vertical database
        self._build_vertical_database(sequences)
        
        # Find frequent 1-sequences
        frequent_1_sequences = self._find_frequent_1_sequences()
        
        # Mine patterns using equivalence classes
        self.frequent_patterns = []
        await self._mine_equivalence_classes(frequent_1_sequences)
        
        return self.frequent_patterns
    
    def _build_vertical_database(self, sequences: List[ExecutionSequence]):
        """Build vertical database representation."""
        self.vertical_db = defaultdict(list)
        
        for seq_id, sequence in enumerate(sequences):
            for pos, step in enumerate(sequence.steps):
                item = f"{step.step_type}:{step.action}"
                self.vertical_db[item].append((seq_id, pos))
    
    def _find_frequent_1_sequences(self) -> List[str]:
        """Find frequent 1-item sequences."""
        frequent_items = []
        
        for item, occurrences in self.vertical_db.items():
            # Count unique sequences
            unique_sequences = len(set(seq_id for seq_id, _ in occurrences))
            if unique_sequences >= self.config.min_support:
                frequent_items.append(item)
        
        return frequent_items
    
    async def _mine_equivalence_classes(self, frequent_items: List[str]):
        """Mine patterns using equivalence classes."""
        # Create initial equivalence classes
        equivalence_classes = []
        
        for i, item1 in enumerate(frequent_items):
            for j, item2 in enumerate(frequent_items):
                if i < j:
                    # Try to join sequences
                    joined_pattern = [item1, item2]
                    support = self._calculate_support(joined_pattern)
                    
                    if support >= self.config.min_support:
                        pattern = SequentialPattern(
                            pattern=joined_pattern,
                            support=support,
                            confidence=support / len(self.vertical_db[item1]),
                            frequency=support,
                            avg_duration=0.0
                        )
                        self.frequent_patterns.append(pattern)
                        equivalence_classes.append(joined_pattern)
        
        # Recursively mine longer patterns
        await self._mine_recursive_spade(equivalence_classes)
    
    async def _mine_recursive_spade(self, patterns: List[List[str]]):
        """Recursively mine longer patterns."""
        if not patterns or len(patterns[0]) >= self.config.max_pattern_length:
            return
        
        new_patterns = []
        
        for i, pattern1 in enumerate(patterns):
            for j, pattern2 in enumerate(patterns):
                if i < j and self._can_join(pattern1, pattern2):
                    joined = self._join_patterns(pattern1, pattern2)
                    support = self._calculate_support(joined)
                    
                    if support >= self.config.min_support:
                        pattern = SequentialPattern(
                            pattern=joined,
                            support=support,
                            confidence=support / self._calculate_support(pattern1),
                            frequency=support,
                            avg_duration=0.0
                        )
                        self.frequent_patterns.append(pattern)
                        new_patterns.append(joined)
        
        if new_patterns:
            await self._mine_recursive_spade(new_patterns)
    
    def _can_join(self, pattern1: List[str], pattern2: List[str]) -> bool:
        """Check if two patterns can be joined."""
        return pattern1[1:] == pattern2[:-1]
    
    def _join_patterns(self, pattern1: List[str], pattern2: List[str]) -> List[str]:
        """Join two patterns."""
        return pattern1 + [pattern2[-1]]
    
    def _calculate_support(self, pattern: List[str]) -> int:
        """Calculate support for a pattern."""
        if len(pattern) == 1:
            return len(set(seq_id for seq_id, _ in self.vertical_db[pattern[0]]))
        
        # For multi-item patterns, need to check sequential occurrence
        supporting_sequences = set()
        
        for seq_id in range(1000):  # Assume max 1000 sequences
            if self._pattern_occurs_in_sequence(pattern, seq_id):
                supporting_sequences.add(seq_id)
        
        return len(supporting_sequences)
    
    def _pattern_occurs_in_sequence(self, pattern: List[str], seq_id: int) -> bool:
        """Check if pattern occurs in given sequence."""
        positions = []
        
        for item in pattern:
            item_positions = [pos for sid, pos in self.vertical_db.get(item, []) if sid == seq_id]
            if not item_positions:
                return False
            positions.append(item_positions)
        
        # Check if there's a valid sequential occurrence
        return self._has_sequential_occurrence(positions)
    
    def _has_sequential_occurrence(self, positions: List[List[int]]) -> bool:
        """Check if positions allow for sequential occurrence."""
        if not positions:
            return False
        
        def backtrack(idx: int, last_pos: int) -> bool:
            if idx == len(positions):
                return True
            
            for pos in positions[idx]:
                if pos > last_pos:
                    if backtrack(idx + 1, pos):
                        return True
            return False
        
        return backtrack(0, -1)


class PatternClusterer:
    """
    Clusters similar behavioral patterns using machine learning techniques.
    """
    
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    async def cluster_patterns(self, patterns: List[SequentialPattern]) -> Dict[int, List[SequentialPattern]]:
        """
        Cluster patterns based on their characteristics.
        
        Args:
            patterns: List of sequential patterns to cluster
            
        Returns:
            Dictionary mapping cluster IDs to patterns
        """
        if not patterns:
            return {}
        
        # Extract features for clustering
        features = self._extract_features(patterns)
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Perform clustering
        cluster_labels = self.kmeans.fit_predict(features_scaled)
        
        # Group patterns by cluster
        clusters = defaultdict(list)
        for pattern, label in zip(patterns, cluster_labels):
            clusters[label].append(pattern)
        
        return dict(clusters)
    
    def _extract_features(self, patterns: List[SequentialPattern]) -> np.ndarray:
        """Extract numerical features from patterns for clustering."""
        features = []
        
        for pattern in patterns:
            feature_vector = [
                len(pattern.pattern),  # Pattern length
                pattern.support,       # Support count
                pattern.confidence,    # Confidence score
                pattern.frequency,     # Frequency
                pattern.avg_duration   # Average duration
            ]
            features.append(feature_vector)
        
        return np.array(features)


class PatternMiner:
    """
    Main pattern mining interface for behavioral analysis.
    """
    
    def __init__(self):
        self.engine = PatternMiningEngine()
    
    async def mine_patterns(self, sequences: List[Any]) -> List[Any]:
        """Mine patterns from execution sequences."""
        try:
            # Convert sequences to the format expected by the engine
            # For now, return empty list as placeholder
            return []
        except Exception as e:
            logger.error(f"Failed to mine patterns: {e}")
            return []
    
    async def detect_anomaly(self, current_sequence: Any, historical_patterns: List[Any]) -> float:
        """Detect anomalies in current sequence compared to historical patterns."""
        try:
            # For now, return a default anomaly score
            return 0.1  # Low anomaly score
        except Exception as e:
            logger.error(f"Failed to detect anomaly: {e}")
            return 0.5  # Medium anomaly score


class PatternMiningEngine:
    """
    Main engine for pattern mining operations.
    
    Coordinates different pattern mining algorithms and provides
    a unified interface for behavioral pattern analysis.
    """
    
    def __init__(self, config: PatternMiningConfig = None):
        self.config = config or PatternMiningConfig()
        self.prefixspan = PrefixSpanMiner(self.config)
        self.spade = SPADEMiner(self.config)
        self.clusterer = PatternClusterer()
    
    async def mine_behavioral_patterns(self, sequences: List[ExecutionSequence]) -> List[BehavioralPattern]:
        """
        Mine behavioral patterns from execution sequences.
        
        Args:
            sequences: List of agent execution sequences
            
        Returns:
            List of discovered behavioral patterns
        """
        # Mine patterns using both algorithms
        prefixspan_patterns = await self.prefixspan.mine_patterns(sequences)
        spade_patterns = await self.spade.mine_patterns(sequences)
        
        # Combine and deduplicate patterns
        all_patterns = self._combine_patterns(prefixspan_patterns, spade_patterns)
        
        # Cluster similar patterns
        pattern_clusters = await self.clusterer.cluster_patterns(all_patterns)
        
        # Convert to BehavioralPattern objects
        behavioral_patterns = []
        for cluster_id, cluster_patterns in pattern_clusters.items():
            behavioral_pattern = self._create_behavioral_pattern(cluster_patterns, cluster_id)
            behavioral_patterns.append(behavioral_pattern)
        
        return behavioral_patterns
    
    def _combine_patterns(self, patterns1: List[SequentialPattern], patterns2: List[SequentialPattern]) -> List[SequentialPattern]:
        """Combine and deduplicate patterns from different algorithms."""
        pattern_dict = {}
        
        # Add patterns from first algorithm
        for pattern in patterns1:
            key = tuple(pattern.pattern)
            pattern_dict[key] = pattern
        
        # Add patterns from second algorithm (merge if duplicate)
        for pattern in patterns2:
            key = tuple(pattern.pattern)
            if key in pattern_dict:
                # Merge statistics
                existing = pattern_dict[key]
                existing.support = max(existing.support, pattern.support)
                existing.confidence = max(existing.confidence, pattern.confidence)
                existing.frequency = max(existing.frequency, pattern.frequency)
            else:
                pattern_dict[key] = pattern
        
        return list(pattern_dict.values())
    
    def _create_behavioral_pattern(self, patterns: List[SequentialPattern], cluster_id: int) -> BehavioralPattern:
        """Create BehavioralPattern from clustered sequential patterns."""
        if not patterns:
            return BehavioralPattern(
                pattern_id=f"cluster_{cluster_id}",
                pattern_name=f"Pattern Cluster {cluster_id}",
                execution_sequences=[],
                frequency=0,
                success_rate=0.0,
                average_duration=0.0,
                common_triggers=[],
                failure_modes=[],
                statistical_significance=0.0
            )
        
        # Aggregate statistics
        total_support = sum(p.support for p in patterns)
        avg_confidence = sum(p.confidence for p in patterns) / len(patterns)
        total_frequency = sum(p.frequency for p in patterns)
        
        # Extract common elements
        common_triggers = self._extract_common_triggers(patterns)
        
        return BehavioralPattern(
            pattern_id=f"cluster_{cluster_id}",
            pattern_name=f"Pattern Cluster {cluster_id}",
            execution_sequences=[],  # Will be populated by caller
            frequency=total_frequency,
            success_rate=avg_confidence,
            average_duration=0.0,  # Will be calculated from sequences
            common_triggers=common_triggers,
            failure_modes=[],  # Will be analyzed separately
            statistical_significance=avg_confidence
        )
    
    def _extract_common_triggers(self, patterns: List[SequentialPattern]) -> List[str]:
        """Extract common triggers from patterns."""
        trigger_counts: Counter[str] = Counter()
        
        for pattern in patterns:
            if pattern.pattern:
                # First item in pattern is often a trigger
                trigger_counts[pattern.pattern[0]] += 1
        
        # Return most common triggers
        return [trigger for trigger, _ in trigger_counts.most_common(5)]