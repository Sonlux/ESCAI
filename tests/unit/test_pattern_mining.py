"""
Unit tests for pattern mining algorithms.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import asyncio

from escai_framework.analytics.pattern_mining import (
    PrefixSpanMiner, SPADEMiner, PatternClusterer, PatternMiningEngine,
    SequentialPattern, PatternMiningConfig
)
from escai_framework.models.behavioral_pattern import ExecutionSequence, ExecutionStep


@pytest.fixture
def sample_execution_sequences():
    """Create sample execution sequences for testing."""
    sequences = []
    
    # Sequence 1: Successful pattern
    steps1 = [
        ExecutionStep(
            step_id="1", step_type="planning", action="analyze_task",
            duration=1.0, success_probability=0.9, context={}, error_message=None
        ),
        ExecutionStep(
            step_id="2", step_type="execution", action="perform_action",
            duration=2.0, success_probability=0.8, context={}, error_message=None
        ),
        ExecutionStep(
            step_id="3", step_type="observation", action="check_result",
            duration=0.5, success_probability=0.95, context={}, error_message=None
        )
    ]
    sequences.append(ExecutionSequence(
        sequence_id="seq1", agent_id="agent1", steps=steps1,
        total_duration=3.5, success_rate=0.88
    ))
    
    # Sequence 2: Similar pattern
    steps2 = [
        ExecutionStep(
            step_id="4", step_type="planning", action="analyze_task",
            duration=1.2, success_probability=0.85, context={}, error_message=None
        ),
        ExecutionStep(
            step_id="5", step_type="execution", action="perform_action",
            duration=1.8, success_probability=0.9, context={}, error_message=None
        ),
        ExecutionStep(
            step_id="6", step_type="observation", action="check_result",
            duration=0.6, success_probability=0.92, context={}, error_message=None
        )
    ]
    sequences.append(ExecutionSequence(
        sequence_id="seq2", agent_id="agent1", steps=steps2,
        total_duration=3.6, success_rate=0.89
    ))
    
    # Sequence 3: Different pattern
    steps3 = [
        ExecutionStep(
            step_id="7", step_type="reasoning", action="think",
            duration=2.0, success_probability=0.7, context={}, error_message=None
        ),
        ExecutionStep(
            step_id="8", step_type="action", action="execute",
            duration=1.5, success_probability=0.8, context={}, error_message=None
        )
    ]
    sequences.append(ExecutionSequence(
        sequence_id="seq3", agent_id="agent2", steps=steps3,
        total_duration=3.5, success_rate=0.75
    ))
    
    return sequences


@pytest.fixture
def pattern_mining_config():
    """Create pattern mining configuration."""
    return PatternMiningConfig(
        min_support=2,
        min_confidence=0.5,
        max_pattern_length=5,
        window_size=10
    )


class TestPrefixSpanMiner:
    """Test PrefixSpan pattern mining algorithm."""
    
    def test_initialization(self, pattern_mining_config):
        """Test PrefixSpan miner initialization."""
        miner = PrefixSpanMiner(pattern_mining_config)
        assert miner.config == pattern_mining_config
        assert miner.frequent_patterns == []
    
    @pytest.mark.asyncio
    async def test_mine_patterns(self, sample_execution_sequences, pattern_mining_config):
        """Test pattern mining functionality."""
        miner = PrefixSpanMiner(pattern_mining_config)
        patterns = await miner.mine_patterns(sample_execution_sequences)
        
        assert isinstance(patterns, list)
        assert len(patterns) > 0
        
        # Check pattern structure
        for pattern in patterns:
            assert isinstance(pattern, SequentialPattern)
            assert isinstance(pattern.pattern, list)
            assert pattern.support >= pattern_mining_config.min_support
            assert 0 <= pattern.confidence <= 1
    
    def test_convert_to_item_sequences(self, sample_execution_sequences, pattern_mining_config):
        """Test conversion of execution sequences to item sequences."""
        miner = PrefixSpanMiner(pattern_mining_config)
        item_sequences = miner._convert_to_item_sequences(sample_execution_sequences)
        
        assert isinstance(item_sequences, list)
        assert len(item_sequences) == len(sample_execution_sequences)
        
        # Check first sequence conversion
        expected_items = ["planning:analyze_task", "execution:perform_action", "observation:check_result"]
        assert item_sequences[0] == expected_items
    
    def test_find_frequent_items(self, pattern_mining_config):
        """Test frequent item identification."""
        miner = PrefixSpanMiner(pattern_mining_config)
        sequences = [
            ["A", "B", "C"],
            ["A", "B", "D"],
            ["A", "C", "D"],
            ["B", "C", "D"]
        ]
        
        frequent_items = miner._find_frequent_items(sequences)
        
        # All items appear at least twice (min_support=2)
        expected_items = {"A", "B", "C", "D"}
        assert set(frequent_items) == expected_items
    
    def test_find_suffix(self, pattern_mining_config):
        """Test suffix finding functionality."""
        miner = PrefixSpanMiner(pattern_mining_config)
        
        # Test exact prefix match
        sequence = ["A", "B", "C", "D"]
        prefix = ["A", "B"]
        suffix = miner._find_suffix(prefix, sequence)
        assert suffix == ["C", "D"]
        
        # Test no match
        prefix = ["X", "Y"]
        suffix = miner._find_suffix(prefix, sequence)
        assert suffix is None
        
        # Test empty prefix
        suffix = miner._find_suffix([], sequence)
        assert suffix == sequence


class TestSPADEMiner:
    """Test SPADE pattern mining algorithm."""
    
    def test_initialization(self, pattern_mining_config):
        """Test SPADE miner initialization."""
        miner = SPADEMiner(pattern_mining_config)
        assert miner.config == pattern_mining_config
        assert miner.vertical_db == {}
        assert miner.frequent_patterns == []
    
    @pytest.mark.asyncio
    async def test_mine_patterns(self, sample_execution_sequences, pattern_mining_config):
        """Test SPADE pattern mining."""
        miner = SPADEMiner(pattern_mining_config)
        patterns = await miner.mine_patterns(sample_execution_sequences)
        
        assert isinstance(patterns, list)
        # May have fewer patterns than PrefixSpan due to different algorithm
        
        for pattern in patterns:
            assert isinstance(pattern, SequentialPattern)
            assert pattern.support >= pattern_mining_config.min_support
    
    def test_build_vertical_database(self, sample_execution_sequences, pattern_mining_config):
        """Test vertical database construction."""
        miner = SPADEMiner(pattern_mining_config)
        miner._build_vertical_database(sample_execution_sequences)
        
        assert len(miner.vertical_db) > 0
        
        # Check structure
        for item, occurrences in miner.vertical_db.items():
            assert isinstance(item, str)
            assert isinstance(occurrences, list)
            for seq_id, pos in occurrences:
                assert isinstance(seq_id, int)
                assert isinstance(pos, int)
    
    def test_find_frequent_1_sequences(self, sample_execution_sequences, pattern_mining_config):
        """Test frequent 1-sequence identification."""
        miner = SPADEMiner(pattern_mining_config)
        miner._build_vertical_database(sample_execution_sequences)
        frequent_items = miner._find_frequent_1_sequences()
        
        assert isinstance(frequent_items, list)
        # Should find items that appear in at least min_support sequences
    
    def test_can_join(self, pattern_mining_config):
        """Test pattern joining capability."""
        miner = SPADEMiner(pattern_mining_config)
        
        # Patterns that can be joined
        pattern1 = ["A", "B"]
        pattern2 = ["B", "C"]
        assert miner._can_join(pattern1, pattern2)
        
        # Patterns that cannot be joined
        pattern1 = ["A", "B"]
        pattern2 = ["C", "D"]
        assert not miner._can_join(pattern1, pattern2)
    
    def test_join_patterns(self, pattern_mining_config):
        """Test pattern joining."""
        miner = SPADEMiner(pattern_mining_config)
        
        pattern1 = ["A", "B"]
        pattern2 = ["B", "C"]
        joined = miner._join_patterns(pattern1, pattern2)
        
        assert joined == ["A", "B", "C"]


class TestPatternClusterer:
    """Test pattern clustering functionality."""
    
    def test_initialization(self):
        """Test clusterer initialization."""
        clusterer = PatternClusterer(n_clusters=3)
        assert clusterer.n_clusters == 3
        assert clusterer.scaler is not None
        assert clusterer.kmeans is not None
    
    @pytest.mark.asyncio
    async def test_cluster_patterns(self):
        """Test pattern clustering."""
        clusterer = PatternClusterer(n_clusters=2)
        
        # Create sample patterns
        patterns = [
            SequentialPattern(["A", "B"], 5, 0.8, 5, 2.0),
            SequentialPattern(["A", "C"], 4, 0.7, 4, 2.1),
            SequentialPattern(["X", "Y"], 3, 0.6, 3, 1.5),
            SequentialPattern(["X", "Z"], 3, 0.65, 3, 1.6)
        ]
        
        clusters = await clusterer.cluster_patterns(patterns)
        
        assert isinstance(clusters, dict)
        assert len(clusters) <= 2  # Should have at most n_clusters
        
        # Check that all patterns are assigned
        total_patterns = sum(len(cluster_patterns) for cluster_patterns in clusters.values())
        assert total_patterns == len(patterns)
    
    @pytest.mark.asyncio
    async def test_cluster_empty_patterns(self):
        """Test clustering with empty pattern list."""
        clusterer = PatternClusterer(n_clusters=2)
        clusters = await clusterer.cluster_patterns([])
        
        assert clusters == {}
    
    def test_extract_features(self):
        """Test feature extraction from patterns."""
        clusterer = PatternClusterer(n_clusters=2)
        
        patterns = [
            SequentialPattern(["A", "B"], 5, 0.8, 5, 2.0),
            SequentialPattern(["X", "Y", "Z"], 3, 0.6, 3, 1.5)
        ]
        
        features = clusterer._extract_features(patterns)
        
        assert features.shape == (2, 5)  # 2 patterns, 5 features each
        
        # Check first pattern features
        assert features[0, 0] == 2  # Pattern length
        assert features[0, 1] == 5  # Support
        assert features[0, 2] == 0.8  # Confidence
        assert features[0, 3] == 5  # Frequency
        assert features[0, 4] == 2.0  # Average duration


class TestPatternMiningEngine:
    """Test the main pattern mining engine."""
    
    def test_initialization(self, pattern_mining_config):
        """Test engine initialization."""
        engine = PatternMiningEngine(pattern_mining_config)
        assert engine.config == pattern_mining_config
        assert isinstance(engine.prefixspan, PrefixSpanMiner)
        assert isinstance(engine.spade, SPADEMiner)
        assert isinstance(engine.clusterer, PatternClusterer)
    
    @pytest.mark.asyncio
    async def test_mine_behavioral_patterns(self, sample_execution_sequences, pattern_mining_config):
        """Test comprehensive behavioral pattern mining."""
        engine = PatternMiningEngine(pattern_mining_config)
        
        with patch.object(engine.prefixspan, 'mine_patterns') as mock_prefixspan, \
             patch.object(engine.spade, 'mine_patterns') as mock_spade, \
             patch.object(engine.clusterer, 'cluster_patterns') as mock_cluster:
            
            # Mock return values
            mock_patterns = [
                SequentialPattern(["A", "B"], 5, 0.8, 5, 2.0),
                SequentialPattern(["X", "Y"], 3, 0.6, 3, 1.5)
            ]
            mock_prefixspan.return_value = mock_patterns
            mock_spade.return_value = mock_patterns
            mock_cluster.return_value = {0: mock_patterns}
            
            behavioral_patterns = await engine.mine_behavioral_patterns(sample_execution_sequences)
            
            assert isinstance(behavioral_patterns, list)
            assert len(behavioral_patterns) > 0
            
            # Verify mocks were called
            mock_prefixspan.assert_called_once()
            mock_spade.assert_called_once()
            mock_cluster.assert_called_once()
    
    def test_combine_patterns(self, pattern_mining_config):
        """Test pattern combination and deduplication."""
        engine = PatternMiningEngine(pattern_mining_config)
        
        patterns1 = [
            SequentialPattern(["A", "B"], 5, 0.8, 5, 2.0),
            SequentialPattern(["X", "Y"], 3, 0.6, 3, 1.5)
        ]
        
        patterns2 = [
            SequentialPattern(["A", "B"], 4, 0.7, 4, 2.1),  # Duplicate
            SequentialPattern(["Z", "W"], 2, 0.5, 2, 1.0)   # New
        ]
        
        combined = engine._combine_patterns(patterns1, patterns2)
        
        # Should have 3 unique patterns (duplicate merged)
        assert len(combined) == 3
        
        # Check that duplicate was merged with higher values
        ab_pattern = next(p for p in combined if p.pattern == ["A", "B"])
        assert ab_pattern.support == 5  # Max of 5 and 4
        assert ab_pattern.confidence == 0.8  # Max of 0.8 and 0.7
    
    def test_extract_common_triggers(self, pattern_mining_config):
        """Test common trigger extraction."""
        engine = PatternMiningEngine(pattern_mining_config)
        
        patterns = [
            SequentialPattern(["trigger1", "action"], 5, 0.8, 5, 2.0),
            SequentialPattern(["trigger1", "other"], 3, 0.6, 3, 1.5),
            SequentialPattern(["trigger2", "action"], 2, 0.5, 2, 1.0)
        ]
        
        triggers = engine._extract_common_triggers(patterns)
        
        assert "trigger1" in triggers  # Most common
        assert len(triggers) <= 5  # Limited to top 5


class TestSequentialPattern:
    """Test SequentialPattern data structure."""
    
    def test_creation(self):
        """Test pattern creation."""
        pattern = SequentialPattern(
            pattern=["A", "B", "C"],
            support=10,
            confidence=0.8,
            frequency=15,
            avg_duration=2.5
        )
        
        assert pattern.pattern == ["A", "B", "C"]
        assert pattern.support == 10
        assert pattern.confidence == 0.8
        assert pattern.frequency == 15
        assert pattern.avg_duration == 2.5


class TestPatternMiningConfig:
    """Test pattern mining configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PatternMiningConfig()
        
        assert config.min_support == 2
        assert config.min_confidence == 0.5
        assert config.max_pattern_length == 10
        assert config.window_size == 100
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = PatternMiningConfig(
            min_support=5,
            min_confidence=0.7,
            max_pattern_length=8,
            window_size=50
        )
        
        assert config.min_support == 5
        assert config.min_confidence == 0.7
        assert config.max_pattern_length == 8
        assert config.window_size == 50


@pytest.mark.integration
class TestPatternMiningIntegration:
    """Integration tests for pattern mining."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_pattern_mining(self, sample_execution_sequences):
        """Test complete pattern mining workflow."""
        config = PatternMiningConfig(min_support=1, min_confidence=0.1)
        engine = PatternMiningEngine(config)
        
        behavioral_patterns = await engine.mine_behavioral_patterns(sample_execution_sequences)
        
        assert isinstance(behavioral_patterns, list)
        # Should find at least some patterns with relaxed thresholds
        
        for pattern in behavioral_patterns:
            assert hasattr(pattern, 'pattern_id')
            assert hasattr(pattern, 'pattern_name')
            assert hasattr(pattern, 'frequency')
            assert hasattr(pattern, 'success_rate')
    
    @pytest.mark.asyncio
    async def test_pattern_mining_performance(self, sample_execution_sequences):
        """Test pattern mining performance with larger dataset."""
        # Create larger dataset
        large_sequences = sample_execution_sequences * 10
        
        config = PatternMiningConfig(min_support=2)
        engine = PatternMiningEngine(config)
        
        import time
        start_time = time.time()
        
        behavioral_patterns = await engine.mine_behavioral_patterns(large_sequences)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert execution_time < 10.0  # 10 seconds
        assert isinstance(behavioral_patterns, list)