"""
Integration tests for PostgreSQL storage operations.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from uuid import uuid4

from escai_framework.storage.database import db_manager, Base
from escai_framework.storage.models import (
    Agent, MonitoringSession, EpistemicStateRecord,
    BehavioralPatternRecord, CausalRelationshipRecord, PredictionRecord
)
from escai_framework.storage.repositories import (
    AgentRepository, MonitoringSessionRepository, EpistemicStateRepository,
    BehavioralPatternRepository, CausalRelationshipRepository, PredictionRepository
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def setup_database():
    """Set up test database."""
    # Initialize database manager with test database
    db_manager.initialize(
        database_url="postgresql://escai:escai@localhost:5432/escai_test",
        async_database_url="postgresql+asyncpg://escai:escai@localhost:5432/escai_test",
        pool_size=5,
        max_overflow=10
    )
    
    # Create tables
    await db_manager.create_tables()
    
    yield db_manager
    
    # Clean up
    await db_manager.drop_tables()
    await db_manager.close()


@pytest.fixture
async def db_session(setup_database):
    """Provide a database session for tests."""
    async with setup_database.get_async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture
def sample_agent_data():
    """Sample agent data for testing."""
    return {
        'agent_id': 'test-agent-001',
        'name': 'Test Agent',
        'framework': 'langchain',
        'version': '1.0.0',
        'description': 'Test agent for integration testing',
        'configuration': {'model': 'gpt-4', 'temperature': 0.7}
    }


@pytest.fixture
def sample_epistemic_state_data():
    """Sample epistemic state data for testing."""
    return {
        'timestamp': datetime.utcnow(),
        'belief_states': [
            {'belief': 'User wants help with coding', 'confidence': 0.9},
            {'belief': 'Task is programming related', 'confidence': 0.8}
        ],
        'knowledge_state': {
            'domain': 'software_development',
            'concepts': ['python', 'testing', 'databases']
        },
        'goal_state': {
            'primary_goal': 'help_user_with_task',
            'sub_goals': ['understand_requirements', 'provide_solution']
        },
        'confidence_level': 0.85,
        'uncertainty_score': 0.15,
        'decision_context': {'context_type': 'coding_assistance'}
    }


class TestAgentRepository:
    """Test AgentRepository operations."""
    
    async def test_create_agent(self, db_session, sample_agent_data):
        """Test creating an agent."""
        repo = AgentRepository()
        agent = await repo.create(db_session, **sample_agent_data)
        
        assert agent.id is not None
        assert agent.agent_id == sample_agent_data['agent_id']
        assert agent.name == sample_agent_data['name']
        assert agent.framework == sample_agent_data['framework']
        assert agent.is_active is True
        assert agent.created_at is not None
    
    async def test_get_by_agent_id(self, db_session, sample_agent_data):
        """Test getting agent by agent_id."""
        repo = AgentRepository()
        created_agent = await repo.create(db_session, **sample_agent_data)
        
        retrieved_agent = await repo.get_by_agent_id(db_session, sample_agent_data['agent_id'])
        
        assert retrieved_agent is not None
        assert retrieved_agent.id == created_agent.id
        assert retrieved_agent.agent_id == sample_agent_data['agent_id']
    
    async def test_get_active_agents(self, db_session, sample_agent_data):
        """Test getting active agents."""
        repo = AgentRepository()
        
        # Create active agent
        active_agent_data = sample_agent_data.copy()
        active_agent_data['agent_id'] = 'active-agent'
        await repo.create(db_session, **active_agent_data)
        
        # Create inactive agent
        inactive_agent_data = sample_agent_data.copy()
        inactive_agent_data['agent_id'] = 'inactive-agent'
        inactive_agent_data['is_active'] = False
        await repo.create(db_session, **inactive_agent_data)
        
        active_agents = await repo.get_active_agents(db_session)
        
        assert len(active_agents) >= 1
        assert all(agent.is_active for agent in active_agents)
    
    async def test_deactivate_agent(self, db_session, sample_agent_data):
        """Test deactivating an agent."""
        repo = AgentRepository()
        await repo.create(db_session, **sample_agent_data)
        
        result = await repo.deactivate_agent(db_session, sample_agent_data['agent_id'])
        assert result is True
        
        agent = await repo.get_by_agent_id(db_session, sample_agent_data['agent_id'])
        assert agent.is_active is False


class TestEpistemicStateRepository:
    """Test EpistemicStateRepository operations."""
    
    async def test_create_epistemic_state(self, db_session, sample_agent_data, sample_epistemic_state_data):
        """Test creating an epistemic state."""
        # Create agent first
        agent_repo = AgentRepository()
        agent = await agent_repo.create(db_session, **sample_agent_data)
        
        # Create epistemic state
        epistemic_repo = EpistemicStateRepository()
        epistemic_data = sample_epistemic_state_data.copy()
        epistemic_data['agent_id'] = agent.id
        
        epistemic_state = await epistemic_repo.create(db_session, **epistemic_data)
        
        assert epistemic_state.id is not None
        assert epistemic_state.agent_id == agent.id
        assert epistemic_state.confidence_level == 0.85
        assert epistemic_state.uncertainty_score == 0.15
    
    async def test_get_latest_by_agent(self, db_session, sample_agent_data, sample_epistemic_state_data):
        """Test getting latest epistemic state for an agent."""
        # Create agent
        agent_repo = AgentRepository()
        agent = await agent_repo.create(db_session, **sample_agent_data)
        
        # Create multiple epistemic states
        epistemic_repo = EpistemicStateRepository()
        
        # Older state
        older_data = sample_epistemic_state_data.copy()
        older_data['agent_id'] = agent.id
        older_data['timestamp'] = datetime.utcnow() - timedelta(hours=1)
        older_data['confidence_level'] = 0.7
        await epistemic_repo.create(db_session, **older_data)
        
        # Newer state
        newer_data = sample_epistemic_state_data.copy()
        newer_data['agent_id'] = agent.id
        newer_data['timestamp'] = datetime.utcnow()
        newer_data['confidence_level'] = 0.9
        await epistemic_repo.create(db_session, **newer_data)
        
        latest_state = await epistemic_repo.get_latest_by_agent(db_session, agent.id)
        
        assert latest_state is not None
        assert latest_state.confidence_level == 0.9
    
    async def test_get_by_time_range(self, db_session, sample_agent_data, sample_epistemic_state_data):
        """Test getting epistemic states by time range."""
        # Create agent
        agent_repo = AgentRepository()
        agent = await agent_repo.create(db_session, **sample_agent_data)
        
        # Create epistemic states at different times
        epistemic_repo = EpistemicStateRepository()
        now = datetime.utcnow()
        
        # State within range
        in_range_data = sample_epistemic_state_data.copy()
        in_range_data['agent_id'] = agent.id
        in_range_data['timestamp'] = now - timedelta(minutes=30)
        await epistemic_repo.create(db_session, **in_range_data)
        
        # State outside range
        out_range_data = sample_epistemic_state_data.copy()
        out_range_data['agent_id'] = agent.id
        out_range_data['timestamp'] = now - timedelta(hours=2)
        await epistemic_repo.create(db_session, **out_range_data)
        
        # Query range
        start_time = now - timedelta(hours=1)
        end_time = now
        
        states = await epistemic_repo.get_by_time_range(
            db_session, agent.id, start_time, end_time
        )
        
        assert len(states) == 1
        assert states[0].timestamp >= start_time
        assert states[0].timestamp <= end_time


class TestBehavioralPatternRepository:
    """Test BehavioralPatternRepository operations."""
    
    async def test_create_behavioral_pattern(self, db_session, sample_agent_data):
        """Test creating a behavioral pattern."""
        # Create agent
        agent_repo = AgentRepository()
        agent = await agent_repo.create(db_session, **sample_agent_data)
        
        # Create behavioral pattern
        pattern_repo = BehavioralPatternRepository()
        pattern_data = {
            'pattern_id': 'pattern-001',
            'agent_id': agent.id,
            'pattern_name': 'Sequential Problem Solving',
            'execution_sequences': [
                ['analyze_problem', 'generate_solution', 'validate_solution'],
                ['analyze_problem', 'research_topic', 'generate_solution']
            ],
            'frequency': 15,
            'success_rate': 0.87,
            'average_duration': 45.5,
            'common_triggers': ['user_question', 'complex_task'],
            'failure_modes': ['timeout', 'insufficient_context'],
            'statistical_significance': 0.95,
            'pattern_type': 'sequential'
        }
        
        pattern = await pattern_repo.create(db_session, **pattern_data)
        
        assert pattern.id is not None
        assert pattern.pattern_id == 'pattern-001'
        assert pattern.agent_id == agent.id
        assert pattern.success_rate == 0.87
        assert pattern.frequency == 15
    
    async def test_get_high_success_patterns(self, db_session, sample_agent_data):
        """Test getting high success rate patterns."""
        # Create agent
        agent_repo = AgentRepository()
        agent = await agent_repo.create(db_session, **sample_agent_data)
        
        # Create patterns with different success rates
        pattern_repo = BehavioralPatternRepository()
        
        # High success pattern
        high_success_data = {
            'pattern_id': 'high-success-pattern',
            'agent_id': agent.id,
            'pattern_name': 'High Success Pattern',
            'execution_sequences': [['step1', 'step2', 'step3']],
            'frequency': 10,
            'success_rate': 0.95,
            'pattern_type': 'sequential'
        }
        await pattern_repo.create(db_session, **high_success_data)
        
        # Low success pattern
        low_success_data = {
            'pattern_id': 'low-success-pattern',
            'agent_id': agent.id,
            'pattern_name': 'Low Success Pattern',
            'execution_sequences': [['step1', 'step2']],
            'frequency': 8,
            'success_rate': 0.6,
            'pattern_type': 'sequential'
        }
        await pattern_repo.create(db_session, **low_success_data)
        
        high_success_patterns = await pattern_repo.get_high_success_patterns(
            db_session, agent.id, min_success_rate=0.8, min_frequency=5
        )
        
        assert len(high_success_patterns) == 1
        assert high_success_patterns[0].success_rate >= 0.8


class TestCausalRelationshipRepository:
    """Test CausalRelationshipRepository operations."""
    
    async def test_create_causal_relationship(self, db_session):
        """Test creating a causal relationship."""
        repo = CausalRelationshipRepository()
        relationship_data = {
            'relationship_id': 'causal-001',
            'cause_event': 'user_asks_question',
            'effect_event': 'agent_analyzes_context',
            'strength': 0.85,
            'confidence': 0.92,
            'delay_ms': 150,
            'evidence': ['temporal_correlation', 'statistical_significance'],
            'statistical_significance': 0.95,
            'causal_mechanism': 'Direct trigger response',
            'analysis_method': 'granger'
        }
        
        relationship = await repo.create(db_session, **relationship_data)
        
        assert relationship.id is not None
        assert relationship.relationship_id == 'causal-001'
        assert relationship.cause_event == 'user_asks_question'
        assert relationship.effect_event == 'agent_analyzes_context'
        assert relationship.strength == 0.85
    
    async def test_get_strong_relationships(self, db_session):
        """Test getting strong causal relationships."""
        repo = CausalRelationshipRepository()
        
        # Strong relationship
        strong_data = {
            'relationship_id': 'strong-causal',
            'cause_event': 'error_occurs',
            'effect_event': 'retry_mechanism_triggered',
            'strength': 0.9,
            'confidence': 0.95,
            'analysis_method': 'structural'
        }
        await repo.create(db_session, **strong_data)
        
        # Weak relationship
        weak_data = {
            'relationship_id': 'weak-causal',
            'cause_event': 'random_event',
            'effect_event': 'unrelated_outcome',
            'strength': 0.3,
            'confidence': 0.5,
            'analysis_method': 'temporal'
        }
        await repo.create(db_session, **weak_data)
        
        strong_relationships = await repo.get_strong_relationships(
            db_session, min_strength=0.7, min_confidence=0.8
        )
        
        assert len(strong_relationships) == 1
        assert strong_relationships[0].strength >= 0.7


class TestPredictionRepository:
    """Test PredictionRepository operations."""
    
    async def test_create_prediction(self, db_session, sample_agent_data):
        """Test creating a prediction."""
        # Create agent
        agent_repo = AgentRepository()
        agent = await agent_repo.create(db_session, **sample_agent_data)
        
        # Create prediction
        prediction_repo = PredictionRepository()
        prediction_data = {
            'prediction_id': 'pred-001',
            'agent_id': agent.id,
            'prediction_type': 'success',
            'predicted_value': {'success_probability': 0.85, 'completion_time': 120},
            'confidence_score': 0.9,
            'risk_factors': ['time_constraint', 'complexity'],
            'recommended_interventions': ['increase_timeout', 'simplify_task'],
            'model_name': 'lstm_predictor',
            'model_version': '1.0.0'
        }
        
        prediction = await prediction_repo.create(db_session, **prediction_data)
        
        assert prediction.id is not None
        assert prediction.prediction_id == 'pred-001'
        assert prediction.agent_id == agent.id
        assert prediction.confidence_score == 0.9
    
    async def test_validate_prediction(self, db_session, sample_agent_data):
        """Test validating a prediction."""
        # Create agent and prediction
        agent_repo = AgentRepository()
        agent = await agent_repo.create(db_session, **sample_agent_data)
        
        prediction_repo = PredictionRepository()
        prediction_data = {
            'prediction_id': 'pred-validate',
            'agent_id': agent.id,
            'prediction_type': 'success',
            'predicted_value': {'success_probability': 0.8},
            'confidence_score': 0.85
        }
        await prediction_repo.create(db_session, **prediction_data)
        
        # Validate prediction
        actual_value = {'actual_success': True}
        accuracy_score = 0.92
        
        validated_prediction = await prediction_repo.validate_prediction(
            db_session, 'pred-validate', actual_value, accuracy_score
        )
        
        assert validated_prediction is not None
        assert validated_prediction.actual_value == actual_value
        assert validated_prediction.accuracy_score == accuracy_score
        assert validated_prediction.validated_at is not None


class TestTransactionHandling:
    """Test transaction handling and error scenarios."""
    
    async def test_transaction_rollback(self, setup_database, sample_agent_data):
        """Test transaction rollback on error."""
        agent_repo = AgentRepository()
        
        try:
            async with setup_database.get_async_session() as session:
                # Create agent
                await agent_repo.create(session, **sample_agent_data)
                
                # Force an error by trying to create duplicate
                await agent_repo.create(session, **sample_agent_data)
                
                # This should not be reached
                assert False, "Should have raised an error"
        except Exception:
            # Expected error due to unique constraint
            pass
        
        # Verify rollback - agent should not exist
        async with setup_database.get_async_session() as session:
            agent = await agent_repo.get_by_agent_id(session, sample_agent_data['agent_id'])
            assert agent is None
    
    async def test_connection_pooling(self, setup_database):
        """Test connection pooling works correctly."""
        agent_repo = AgentRepository()
        
        # Create multiple concurrent sessions
        async def create_agent(agent_id: str):
            async with setup_database.get_async_session() as session:
                return await agent_repo.create(
                    session,
                    agent_id=agent_id,
                    name=f'Agent {agent_id}',
                    framework='test'
                )
        
        # Run multiple concurrent operations
        tasks = [create_agent(f'agent-{i}') for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        assert all(result.id is not None for result in results)