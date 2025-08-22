"""
Unit tests for the EpistemicExtractor class.

This module tests all extraction methods with sample agent logs to ensure
proper belief extraction, confidence parsing, knowledge graph construction,
goal tracking, and uncertainty quantification.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import List

from escai_framework.core.epistemic_extractor import EpistemicExtractor
from escai_framework.models.epistemic_state import (
    EpistemicState, BeliefState, KnowledgeState, GoalState,
    BeliefType, GoalStatus
)
from escai_framework.instrumentation.events import AgentEvent, EventType


class TestEpistemicExtractor:
    """Test suite for EpistemicExtractor class."""
    
    @pytest.fixture
    def extractor(self):
        """Create an EpistemicExtractor instance for testing."""
        return EpistemicExtractor()
    
    @pytest.fixture
    def sample_events(self):
        """Create sample agent events for testing."""
        base_time = datetime.now()
        
        return [
            AgentEvent(
                event_id="test_1",
                agent_id="test_agent",
                session_id="test_session",
                timestamp=base_time,
                event_type=EventType.BELIEF_UPDATE,
                framework="test",
                component="test_component",
                message="I believe the user wants to book a flight with 85% confidence",
                data={"content": "Based on the conversation, I'm confident this is a booking request"}
            ),
            AgentEvent(
                event_id="test_2",
                agent_id="test_agent",
                session_id="test_session",
                timestamp=base_time + timedelta(seconds=1),
                event_type=EventType.KNOWLEDGE_UPDATE,
                framework="test",
                component="test_component",
                message="Fact: Flight prices are higher on weekends",
                data={"content": "Rule: If booking on weekend then expect 20% price increase"}
            ),
            AgentEvent(
                event_id="test_3",
                agent_id="test_agent",
                session_id="test_session",
                timestamp=base_time + timedelta(seconds=2),
                event_type=EventType.TASK_START,
                framework="test",
                component="test_component",
                message="Goal: Book a flight from NYC to LAX",
                data={"content": "Priority: high priority task"}
            )
        ]
    
    @pytest.mark.asyncio
    async def test_extract_beliefs(self, extractor, sample_events):
        """Test belief extraction from agent logs."""
        beliefs = await extractor.extract_beliefs(sample_events)
        
        assert len(beliefs) > 0
        assert isinstance(beliefs[0], BeliefState)
        assert beliefs[0].confidence > 0.0
        assert beliefs[0].content is not None
    
    @pytest.mark.asyncio
    async def test_extract_knowledge(self, extractor, sample_events):
        """Test knowledge extraction from agent logs."""
        knowledge = await extractor.extract_knowledge(sample_events)
        
        assert isinstance(knowledge, KnowledgeState)
        assert len(knowledge.facts) > 0
        assert len(knowledge.rules) > 0
        assert knowledge.confidence_score >= 0.0
    
    @pytest.mark.asyncio
    async def test_extract_goals(self, extractor, sample_events):
        """Test goal extraction from agent logs."""
        goals = await extractor.extract_goals(sample_events)
        
        assert len(goals) > 0
        assert isinstance(goals[0], GoalState)
        assert goals[0].description is not None
        assert goals[0].status in [GoalStatus.ACTIVE, GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.SUSPENDED]
    
    @pytest.mark.asyncio
    async def test_calculate_confidence(self, extractor):
        """Test confidence calculation from text."""
        # Test explicit confidence
        confidence1 = await extractor.calculate_confidence("I am 90% confident this is correct")
        assert confidence1 == 0.9
        
        # Test linguistic confidence
        confidence2 = await extractor.calculate_confidence("I definitely know this is true")
        assert confidence2 > 0.8
        
        # Test low confidence
        confidence3 = await extractor.calculate_confidence("Maybe this could be right")
        assert confidence3 < 0.5
    
    @pytest.mark.asyncio
    async def test_quantify_uncertainty(self, extractor):
        """Test uncertainty quantification."""
        beliefs = [
            BeliefState(
                content="Test belief 1",
                belief_type=BeliefType.FACTUAL,
                confidence=0.9,
                evidence=[],
                timestamp=datetime.now(),
                source="test"
            ),
            BeliefState(
                content="Test belief 2",
                belief_type=BeliefType.PROBABILISTIC,
                confidence=0.3,
                evidence=[],
                timestamp=datetime.now(),
                source="test"
            )
        ]
        
        uncertainty = await extractor.quantify_uncertainty(beliefs)
        assert 0.0 <= uncertainty <= 1.0
    
    def test_confidence_score_extraction(self, extractor):
        """Test regex-based confidence score extraction."""
        # Test percentage format
        score1 = extractor._extract_confidence_score("confidence: 85%")
        assert score1 == 0.85
        
        # Test decimal format
        score2 = extractor._extract_confidence_score("I am 0.7 confident")
        assert score2 == 0.7
        
        # Test no explicit confidence
        score3 = extractor._extract_confidence_score("I think this is right")
        assert score3 == 0.7  # Default for belief statements
    
    def test_belief_type_classification(self, extractor):
        """Test belief type classification."""
        # Test factual belief
        belief_type1 = extractor._classify_belief_type("This is definitely true")
        assert belief_type1 == BeliefType.FACTUAL
        
        # Test probabilistic belief
        belief_type2 = extractor._classify_belief_type("This is probably correct")
        assert belief_type2 == BeliefType.PROBABILISTIC
        
        # Test conditional belief
        belief_type3 = extractor._classify_belief_type("If this happens then that will occur")
        assert belief_type3 == BeliefType.CONDITIONAL
        
        # Test temporal belief
        belief_type4 = extractor._classify_belief_type("This will happen tomorrow")
        assert belief_type4 == BeliefType.TEMPORAL
    
    def test_goal_status_classification(self, extractor):
        """Test goal status classification."""
        # Test active goal
        status1 = extractor._classify_goal_status("Currently working on this task", EventType.TASK_START)
        assert status1 == GoalStatus.ACTIVE
        
        # Test completed goal
        status2 = extractor._classify_goal_status("Task completed successfully", EventType.TASK_COMPLETE)
        assert status2 == GoalStatus.COMPLETED
        
        # Test failed goal
        status3 = extractor._classify_goal_status("Unable to complete task", EventType.TASK_FAIL)
        assert status3 == GoalStatus.FAILED
    
    def test_knowledge_graph_building(self, extractor):
        """Test knowledge graph construction."""
        knowledge_state = KnowledgeState(
            facts=["The sky is blue", "Water boils at 100C"],
            rules=["If temperature > 100C then water boils"],
            concepts={"Water": {"mentions": 2}, "Temperature": {"mentions": 1}},
            relationships=[
                {"subject": "Water", "predicate": "has_property", "object": "Temperature"}
            ],
            confidence_score=0.8
        )
        
        # This will return None if NetworkX is not available, which is fine for testing
        graph = asyncio.run(extractor.build_knowledge_graph(knowledge_state))
        
        # If NetworkX is available, test the graph structure
        if graph is not None:
            assert graph.number_of_nodes() > 0
            assert "Water" in graph.nodes()
            assert "Temperature" in graph.nodes()
    
    def test_linguistic_confidence_analysis(self, extractor):
        """Test linguistic confidence analysis."""
        # High confidence text
        high_conf = extractor._analyze_linguistic_confidence("I definitely know this is absolutely correct")
        assert high_conf > 0.8
        
        # Low confidence text
        low_conf = extractor._analyze_linguistic_confidence("Maybe this could possibly be right")
        assert low_conf < 0.5
        
        # Medium confidence text
        med_conf = extractor._analyze_linguistic_confidence("I think this is probably correct")
        assert 0.4 < med_conf < 0.8
    
    def test_goal_priority_extraction(self, extractor):
        """Test goal priority extraction."""
        # Explicit priority
        priority1 = extractor._extract_goal_priority("Priority: 8")
        assert priority1 == 8
        
        # High priority keywords
        priority2 = extractor._extract_goal_priority("This is urgent and high priority")
        assert priority2 >= 8
        
        # Low priority keywords
        priority3 = extractor._extract_goal_priority("This is low priority")
        assert priority3 <= 3
        
        # Default priority
        priority4 = extractor._extract_goal_priority("Just a regular task")
        assert priority4 == 5
    
    @pytest.mark.asyncio
    async def test_empty_input_handling(self, extractor):
        """Test handling of empty inputs."""
        # Empty events list
        beliefs = await extractor.extract_beliefs([])
        assert beliefs == []
        
        knowledge = await extractor.extract_knowledge([])
        assert isinstance(knowledge, KnowledgeState)
        assert len(knowledge.facts) == 0
        
        goals = await extractor.extract_goals([])
        assert goals == []
        
        # Empty content
        confidence = await extractor.calculate_confidence("")
        assert confidence == 0.0
        
        uncertainty = await extractor.quantify_uncertainty([])
        assert uncertainty == 1.0  # Maximum uncertainty with no beliefs
    
    def test_error_handling(self, extractor):
        """Test error handling in extraction methods."""
        # Test with malformed event
        malformed_event = AgentEvent(
            event_id="malformed",
            agent_id="test",
            session_id="test",
            timestamp=datetime.now(),
            event_type=EventType.BELIEF_UPDATE,
            framework="test",
            component="test",
            message=None,  # None message
            data={}
        )
        
        # Should not raise exception
        beliefs = asyncio.run(extractor.extract_beliefs([malformed_event]))
        assert isinstance(beliefs, list)
    
    @patch('escai_framework.core.epistemic_extractor.pipeline')
    def test_model_initialization_fallback(self, mock_pipeline):
        """Test graceful fallback when models are not available."""
        mock_pipeline.side_effect = Exception("Model not available")
        
        # Should not raise exception
        extractor = EpistemicExtractor()
        assert extractor.sentiment_analyzer is None
    
    def test_shannon_entropy_calculation(self, extractor):
        """Test Shannon entropy calculation for uncertainty."""
        # Test with uniform distribution (high entropy)
        confidences = [0.5, 0.5, 0.5, 0.5]
        entropy = extractor._calculate_shannon_entropy(confidences)
        assert entropy > 0
        
        # Test with single value (low entropy)
        confidences_single = [1.0]
        entropy_single = extractor._calculate_shannon_entropy(confidences_single)
        assert entropy_single == 0.0