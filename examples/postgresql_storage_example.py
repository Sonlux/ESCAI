"""
Example demonstrating PostgreSQL storage functionality in ESCAI Framework.

This example shows how to:
1. Initialize the database manager
2. Create agents and monitoring sessions
3. Store epistemic states and behavioral patterns
4. Query data using repositories

Note: This example requires a running PostgreSQL instance.
"""

import asyncio
import os
from datetime import datetime, timedelta
from uuid import uuid4

from escai_framework.storage import (
    db_manager, AgentRepository, MonitoringSessionRepository,
    EpistemicStateRepository, BehavioralPatternRepository,
    CausalRelationshipRepository, PredictionRepository
)


async def setup_database():
    """Initialize database connection and create tables."""
    print("Setting up database connection...")
    
    # Initialize database manager
    db_manager.initialize(
        database_url=os.getenv('ESCAI_DATABASE_URL', 'postgresql://escai:escai@localhost:5432/escai'),
        async_database_url=os.getenv('ESCAI_ASYNC_DATABASE_URL', 'postgresql+asyncpg://escai:escai@localhost:5432/escai')
    )
    
    # Create tables
    await db_manager.create_tables()
    print("Database tables created successfully!")


async def create_sample_agent():
    """Create a sample agent."""
    print("\nCreating sample agent...")
    
    agent_repo = AgentRepository()
    
    async with db_manager.get_async_session() as session:
        agent = await agent_repo.create(
            session,
            agent_id='demo-agent-001',
            name='Demo LangChain Agent',
            framework='langchain',
            version='1.0.0',
            description='Demonstration agent for PostgreSQL storage',
            configuration={
                'model': 'gpt-4',
                'temperature': 0.7,
                'max_tokens': 1000
            }
        )
        
        print(f"Created agent: {agent.name} (ID: {agent.agent_id})")
        return agent


async def create_monitoring_session(agent):
    """Create a monitoring session for the agent."""
    print("\nCreating monitoring session...")
    
    session_repo = MonitoringSessionRepository()
    
    async with db_manager.get_async_session() as session:
        monitoring_session = await session_repo.create(
            session,
            session_id=f'session-{uuid4()}',
            agent_id=agent.id,
            status='active',
            configuration={
                'monitoring_level': 'detailed',
                'capture_epistemic_states': True,
                'capture_patterns': True
            },
            session_metadata={
                'environment': 'demo',
                'purpose': 'storage_example'
            }
        )
        
        print(f"Created monitoring session: {monitoring_session.session_id}")
        return monitoring_session


async def store_epistemic_states(agent, monitoring_session):
    """Store sample epistemic states."""
    print("\nStoring epistemic states...")
    
    epistemic_repo = EpistemicStateRepository()
    
    async with db_manager.get_async_session() as session:
        # Create multiple epistemic states over time
        base_time = datetime.utcnow()
        
        states_data = [
            {
                'timestamp': base_time - timedelta(minutes=10),
                'belief_states': [
                    {'belief': 'User wants help with coding', 'confidence': 0.9},
                    {'belief': 'Task involves Python programming', 'confidence': 0.8}
                ],
                'knowledge_state': {
                    'domain': 'software_development',
                    'concepts': ['python', 'databases', 'sqlalchemy'],
                    'expertise_level': 'intermediate'
                },
                'goal_state': {
                    'primary_goal': 'help_user_with_database_setup',
                    'sub_goals': ['understand_requirements', 'provide_code_examples']
                },
                'confidence_level': 0.85,
                'uncertainty_score': 0.15
            },
            {
                'timestamp': base_time - timedelta(minutes=5),
                'belief_states': [
                    {'belief': 'User understands basic concepts', 'confidence': 0.7},
                    {'belief': 'Need to provide more detailed examples', 'confidence': 0.9}
                ],
                'knowledge_state': {
                    'domain': 'software_development',
                    'concepts': ['python', 'databases', 'sqlalchemy', 'postgresql'],
                    'expertise_level': 'intermediate'
                },
                'goal_state': {
                    'primary_goal': 'help_user_with_database_setup',
                    'sub_goals': ['provide_detailed_examples', 'explain_best_practices']
                },
                'confidence_level': 0.9,
                'uncertainty_score': 0.1
            },
            {
                'timestamp': base_time,
                'belief_states': [
                    {'belief': 'User is making good progress', 'confidence': 0.95},
                    {'belief': 'Task is nearly complete', 'confidence': 0.8}
                ],
                'knowledge_state': {
                    'domain': 'software_development',
                    'concepts': ['python', 'databases', 'sqlalchemy', 'postgresql', 'async'],
                    'expertise_level': 'advanced'
                },
                'goal_state': {
                    'primary_goal': 'help_user_with_database_setup',
                    'sub_goals': ['finalize_implementation', 'provide_testing_guidance']
                },
                'confidence_level': 0.95,
                'uncertainty_score': 0.05
            }
        ]
        
        for state_data in states_data:
            state_data.update({
                'agent_id': agent.id,
                'session_id': monitoring_session.session_id,
                'decision_context': {'interaction_type': 'coding_assistance'},
                'extraction_method': 'nlp_analysis',
                'processing_time_ms': 50
            })
            
            epistemic_state = await epistemic_repo.create(session, **state_data)
            print(f"Stored epistemic state at {epistemic_state.timestamp} (confidence: {epistemic_state.confidence_level})")


async def store_behavioral_patterns(agent):
    """Store sample behavioral patterns."""
    print("\nStoring behavioral patterns...")
    
    pattern_repo = BehavioralPatternRepository()
    
    async with db_manager.get_async_session() as session:
        patterns_data = [
            {
                'pattern_id': 'sequential-problem-solving',
                'pattern_name': 'Sequential Problem Solving',
                'execution_sequences': [
                    ['analyze_problem', 'research_solution', 'generate_code', 'validate_solution'],
                    ['analyze_problem', 'break_down_steps', 'implement_solution', 'test_solution'],
                    ['understand_context', 'research_best_practices', 'provide_examples', 'explain_concepts']
                ],
                'frequency': 25,
                'success_rate': 0.88,
                'average_duration': 120.5,
                'common_triggers': ['complex_coding_question', 'multi_step_problem'],
                'failure_modes': ['insufficient_context', 'time_constraint'],
                'statistical_significance': 0.95,
                'pattern_type': 'sequential',
                'confidence_score': 0.9
            },
            {
                'pattern_id': 'iterative-refinement',
                'pattern_name': 'Iterative Refinement',
                'execution_sequences': [
                    ['initial_solution', 'get_feedback', 'refine_solution', 'validate_improvement'],
                    ['draft_response', 'review_accuracy', 'improve_clarity', 'finalize_response']
                ],
                'frequency': 18,
                'success_rate': 0.92,
                'average_duration': 85.3,
                'common_triggers': ['user_feedback', 'unclear_requirements'],
                'failure_modes': ['infinite_loop', 'over_optimization'],
                'statistical_significance': 0.89,
                'pattern_type': 'iterative',
                'confidence_score': 0.85
            }
        ]
        
        for pattern_data in patterns_data:
            pattern_data['agent_id'] = agent.id
            pattern = await pattern_repo.create(session, **pattern_data)
            print(f"Stored behavioral pattern: {pattern.pattern_name} (success rate: {pattern.success_rate})")


async def store_causal_relationships():
    """Store sample causal relationships."""
    print("\nStoring causal relationships...")
    
    causal_repo = CausalRelationshipRepository()
    
    async with db_manager.get_async_session() as session:
        relationships_data = [
            {
                'relationship_id': 'user-question-analysis',
                'cause_event': 'user_asks_complex_question',
                'effect_event': 'agent_increases_analysis_depth',
                'strength': 0.85,
                'confidence': 0.92,
                'delay_ms': 200,
                'evidence': ['temporal_correlation', 'statistical_significance', 'domain_knowledge'],
                'statistical_significance': 0.95,
                'causal_mechanism': 'Complexity detection triggers deeper analysis mode',
                'analysis_method': 'granger'
            },
            {
                'relationship_id': 'context-clarity-success',
                'cause_event': 'sufficient_context_provided',
                'effect_event': 'high_success_probability',
                'strength': 0.78,
                'confidence': 0.88,
                'delay_ms': 50,
                'evidence': ['correlation_analysis', 'success_rate_comparison'],
                'statistical_significance': 0.91,
                'causal_mechanism': 'Better context leads to more accurate responses',
                'analysis_method': 'structural'
            }
        ]
        
        for rel_data in relationships_data:
            relationship = await causal_repo.create(session, **rel_data)
            print(f"Stored causal relationship: {relationship.cause_event} -> {relationship.effect_event} (strength: {relationship.strength})")


async def store_predictions(agent, monitoring_session):
    """Store sample predictions."""
    print("\nStoring predictions...")
    
    prediction_repo = PredictionRepository()
    
    async with db_manager.get_async_session() as session:
        predictions_data = [
            {
                'prediction_id': f'pred-success-{uuid4()}',
                'prediction_type': 'success',
                'predicted_value': {
                    'success_probability': 0.89,
                    'confidence_interval': [0.82, 0.96]
                },
                'confidence_score': 0.91,
                'risk_factors': ['time_constraint', 'complexity_level'],
                'recommended_interventions': ['provide_examples', 'break_down_steps'],
                'model_name': 'lstm_success_predictor',
                'model_version': '1.2.0',
                'feature_importance': {
                    'context_clarity': 0.35,
                    'user_expertise': 0.28,
                    'task_complexity': 0.22,
                    'historical_success': 0.15
                }
            },
            {
                'prediction_id': f'pred-completion-{uuid4()}',
                'prediction_type': 'completion_time',
                'predicted_value': {
                    'estimated_seconds': 180,
                    'confidence_interval': [150, 220]
                },
                'confidence_score': 0.84,
                'risk_factors': ['user_response_delay', 'clarification_needed'],
                'recommended_interventions': ['set_expectations', 'provide_progress_updates'],
                'model_name': 'time_estimation_model',
                'model_version': '1.0.0'
            }
        ]
        
        for pred_data in predictions_data:
            pred_data.update({
                'agent_id': agent.id,
                'session_id': monitoring_session.session_id,
                'target_time': datetime.utcnow() + timedelta(minutes=5)
            })
            
            prediction = await prediction_repo.create(session, **pred_data)
            print(f"Stored prediction: {prediction.prediction_type} (confidence: {prediction.confidence_score})")


async def query_data(agent):
    """Demonstrate querying stored data."""
    print("\nQuerying stored data...")
    
    async with db_manager.get_async_session() as session:
        # Query epistemic states
        epistemic_repo = EpistemicStateRepository()
        recent_states = await epistemic_repo.get_recent_states(session, agent.id, hours=1)
        print(f"Found {len(recent_states)} recent epistemic states")
        
        if recent_states:
            latest_state = recent_states[-1]
            print(f"Latest confidence level: {latest_state.confidence_level}")
        
        # Query behavioral patterns
        pattern_repo = BehavioralPatternRepository()
        high_success_patterns = await pattern_repo.get_high_success_patterns(
            session, agent.id, min_success_rate=0.8
        )
        print(f"Found {len(high_success_patterns)} high-success patterns")
        
        for pattern in high_success_patterns:
            print(f"  - {pattern.pattern_name}: {pattern.success_rate:.2f} success rate")
        
        # Query causal relationships
        causal_repo = CausalRelationshipRepository()
        strong_relationships = await causal_repo.get_strong_relationships(
            session, min_strength=0.7, min_confidence=0.8
        )
        print(f"Found {len(strong_relationships)} strong causal relationships")
        
        for rel in strong_relationships:
            print(f"  - {rel.cause_event} -> {rel.effect_event} (strength: {rel.strength:.2f})")
        
        # Query predictions
        prediction_repo = PredictionRepository()
        recent_predictions = await prediction_repo.get_recent_predictions(
            session, agent.id, hours=1
        )
        print(f"Found {len(recent_predictions)} recent predictions")
        
        for pred in recent_predictions:
            print(f"  - {pred.prediction_type}: {pred.confidence_score:.2f} confidence")


async def cleanup_database():
    """Clean up database connections."""
    print("\nCleaning up database connections...")
    await db_manager.close()
    print("Database connections closed.")


async def main():
    """Main example function."""
    try:
        # Setup
        await setup_database()
        
        # Create sample data
        agent = await create_sample_agent()
        monitoring_session = await create_monitoring_session(agent)
        
        # Store different types of data
        await store_epistemic_states(agent, monitoring_session)
        await store_behavioral_patterns(agent)
        await store_causal_relationships()
        await store_predictions(agent, monitoring_session)
        
        # Query and display data
        await query_data(agent)
        
        print("\n✅ PostgreSQL storage example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running example: {e}")
        raise
    finally:
        await cleanup_database()


if __name__ == "__main__":
    print("ESCAI Framework - PostgreSQL Storage Example")
    print("=" * 50)
    
    # Check if database URL is configured
    db_url = os.getenv('ESCAI_DATABASE_URL')
    if not db_url:
        print("⚠️  Warning: ESCAI_DATABASE_URL not set, using default localhost connection")
        print("   Make sure PostgreSQL is running on localhost:5432 with database 'escai'")
        print("   and user 'escai' with password 'escai'")
        print()
    
    asyncio.run(main())