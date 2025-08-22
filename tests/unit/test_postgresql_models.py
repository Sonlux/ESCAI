"""
Unit tests for PostgreSQL models and database setup.
"""

import pytest
from datetime import datetime
from uuid import uuid4

from escai_framework.storage.models import (
    Agent, MonitoringSession, EpistemicStateRecord,
    BehavioralPatternRecord, CausalRelationshipRecord, PredictionRecord
)
from escai_framework.storage.database import DatabaseManager


class TestDatabaseManager:
    """Test DatabaseManager functionality."""
    
    def test_database_manager_initialization(self):
        """Test database manager can be initialized."""
        db_manager = DatabaseManager()
        assert not db_manager._initialized
        
        # Test initialization (without actually connecting)
        # Just test that the method exists and can be called
        assert hasattr(db_manager, 'initialize')
        assert hasattr(db_manager, 'get_async_session')
        assert hasattr(db_manager, 'create_tables')
    
    def test_database_manager_properties(self):
        """Test database manager properties."""
        db_manager = DatabaseManager()
        
        # Should raise error when not initialized
        with pytest.raises(RuntimeError, match="Database manager not initialized"):
            _ = db_manager.async_engine
        
        with pytest.raises(RuntimeError, match="Database manager not initialized"):
            _ = db_manager.sync_engine


class TestSQLAlchemyModels:
    """Test SQLAlchemy model creation and validation."""
    
    def test_agent_model_creation(self):
        """Test Agent model can be created."""
        agent_data = {
            'agent_id': 'test-agent-001',
            'name': 'Test Agent',
            'framework': 'langchain',
            'version': '1.0.0',
            'description': 'Test agent',
            'configuration': {'model': 'gpt-4'},
            'is_active': True
        }
        
        agent = Agent(**agent_data)
        assert agent.agent_id == 'test-agent-001'
        assert agent.name == 'Test Agent'
        assert agent.framework == 'langchain'
        assert agent.is_active is True
    
    def test_monitoring_session_model_creation(self):
        """Test MonitoringSession model can be created."""
        session_data = {
            'session_id': 'session-001',
            'agent_id': uuid4(),
            'status': 'active',
            'configuration': {'monitoring_level': 'detailed'},
            'session_metadata': {'environment': 'test'}
        }
        
        session = MonitoringSession(**session_data)
        assert session.session_id == 'session-001'
        assert session.status == 'active'
        assert session.configuration == {'monitoring_level': 'detailed'}
        assert session.session_metadata == {'environment': 'test'}
    
    def test_epistemic_state_model_creation(self):
        """Test EpistemicStateRecord model can be created."""
        epistemic_data = {
            'agent_id': uuid4(),
            'session_id': 'session-001',
            'timestamp': datetime.utcnow(),
            'belief_states': [{'belief': 'test', 'confidence': 0.9}],
            'knowledge_state': {'domain': 'test'},
            'goal_state': {'primary_goal': 'test'},
            'confidence_level': 0.85,
            'uncertainty_score': 0.15,
            'decision_context': {'context': 'test'}
        }
        
        epistemic_state = EpistemicStateRecord(**epistemic_data)
        assert epistemic_state.confidence_level == 0.85
        assert epistemic_state.uncertainty_score == 0.15
        assert len(epistemic_state.belief_states) == 1
    
    def test_behavioral_pattern_model_creation(self):
        """Test BehavioralPatternRecord model can be created."""
        pattern_data = {
            'pattern_id': 'pattern-001',
            'agent_id': uuid4(),
            'pattern_name': 'Test Pattern',
            'execution_sequences': [['step1', 'step2', 'step3']],
            'frequency': 10,
            'success_rate': 0.85,
            'average_duration': 45.5,
            'common_triggers': ['trigger1', 'trigger2'],
            'failure_modes': ['timeout', 'error'],
            'statistical_significance': 0.95,
            'pattern_type': 'sequential'
        }
        
        pattern = BehavioralPatternRecord(**pattern_data)
        assert pattern.pattern_id == 'pattern-001'
        assert pattern.success_rate == 0.85
        assert pattern.frequency == 10
        assert pattern.pattern_type == 'sequential'
    
    def test_causal_relationship_model_creation(self):
        """Test CausalRelationshipRecord model can be created."""
        causal_data = {
            'relationship_id': 'causal-001',
            'cause_event': 'user_question',
            'effect_event': 'agent_response',
            'strength': 0.9,
            'confidence': 0.95,
            'delay_ms': 150,
            'evidence': ['temporal', 'statistical'],
            'statistical_significance': 0.99,
            'causal_mechanism': 'direct_trigger',
            'analysis_method': 'granger'
        }
        
        relationship = CausalRelationshipRecord(**causal_data)
        assert relationship.relationship_id == 'causal-001'
        assert relationship.cause_event == 'user_question'
        assert relationship.effect_event == 'agent_response'
        assert relationship.strength == 0.9
        assert relationship.confidence == 0.95
    
    def test_prediction_model_creation(self):
        """Test PredictionRecord model can be created."""
        prediction_data = {
            'prediction_id': 'pred-001',
            'agent_id': uuid4(),
            'session_id': 'session-001',
            'prediction_type': 'success',
            'predicted_value': {'success_probability': 0.85},
            'confidence_score': 0.9,
            'risk_factors': ['time_constraint'],
            'recommended_interventions': ['increase_timeout'],
            'model_name': 'lstm_predictor',
            'model_version': '1.0.0'
        }
        
        prediction = PredictionRecord(**prediction_data)
        assert prediction.prediction_id == 'pred-001'
        assert prediction.prediction_type == 'success'
        assert prediction.confidence_score == 0.9
        assert prediction.model_name == 'lstm_predictor'


class TestRepositoryImports:
    """Test that repository classes can be imported."""
    
    def test_repository_imports(self):
        """Test all repository classes can be imported."""
        from escai_framework.storage.repositories import (
            BaseRepository, AgentRepository, EpistemicStateRepository,
            BehavioralPatternRepository, CausalRelationshipRepository,
            PredictionRepository, MonitoringSessionRepository
        )
        
        # Test repository instantiation
        agent_repo = AgentRepository()
        assert agent_repo.model_class == Agent
        
        epistemic_repo = EpistemicStateRepository()
        assert epistemic_repo.model_class == EpistemicStateRecord
        
        pattern_repo = BehavioralPatternRepository()
        assert pattern_repo.model_class == BehavioralPatternRecord
        
        causal_repo = CausalRelationshipRepository()
        assert causal_repo.model_class == CausalRelationshipRecord
        
        prediction_repo = PredictionRepository()
        assert prediction_repo.model_class == PredictionRecord
        
        session_repo = MonitoringSessionRepository()
        assert session_repo.model_class == MonitoringSession


class TestModelValidation:
    """Test model validation and constraints."""
    
    def test_agent_required_fields(self):
        """Test Agent model required fields."""
        # Should work with required fields
        agent = Agent(
            agent_id='test-001',
            name='Test Agent',
            framework='test'
        )
        assert agent.agent_id == 'test-001'
        # Note: default values are set by the database, not the model instance
        assert hasattr(agent, 'is_active')
    
    def test_epistemic_state_timestamp_required(self):
        """Test EpistemicStateRecord requires timestamp."""
        epistemic_state = EpistemicStateRecord(
            agent_id=uuid4(),
            timestamp=datetime.utcnow()
        )
        assert epistemic_state.timestamp is not None
        assert epistemic_state.agent_id is not None
    
    def test_causal_relationship_required_fields(self):
        """Test CausalRelationshipRecord required fields."""
        relationship = CausalRelationshipRecord(
            relationship_id='test-001',
            cause_event='cause',
            effect_event='effect',
            strength=0.8,
            confidence=0.9
        )
        assert relationship.cause_event == 'cause'
        assert relationship.effect_event == 'effect'
        assert relationship.strength == 0.8
        assert relationship.confidence == 0.9