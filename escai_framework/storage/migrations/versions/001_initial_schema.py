"""Initial database schema

Revision ID: 001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create agents table
    op.create_table('agents',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('agent_id', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('framework', sa.String(length=100), nullable=False),
        sa.Column('version', sa.String(length=50), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('configuration', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_agent_active', 'agents', ['is_active'])
    op.create_index('idx_agent_framework', 'agents', ['framework'])
    op.create_index(op.f('ix_agents_agent_id'), 'agents', ['agent_id'], unique=True)

    # Create monitoring_sessions table
    op.create_table('monitoring_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('session_id', sa.String(length=255), nullable=False),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=False),
        sa.Column('ended_at', sa.DateTime(), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('configuration', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('session_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_session_started', 'monitoring_sessions', ['started_at'])
    op.create_index('idx_session_status', 'monitoring_sessions', ['status'])
    op.create_index(op.f('ix_monitoring_sessions_session_id'), 'monitoring_sessions', ['session_id'], unique=True)

    # Create epistemic_states table
    op.create_table('epistemic_states',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('session_id', sa.String(length=255), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('belief_states', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('knowledge_state', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('goal_state', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('confidence_level', sa.Float(), nullable=True),
        sa.Column('uncertainty_score', sa.Float(), nullable=True),
        sa.Column('decision_context', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('extraction_method', sa.String(length=100), nullable=True),
        sa.Column('processing_time_ms', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ),
        sa.ForeignKeyConstraint(['session_id'], ['monitoring_sessions.session_id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_epistemic_agent_time', 'epistemic_states', ['agent_id', 'timestamp'])
    op.create_index('idx_epistemic_session', 'epistemic_states', ['session_id'])
    op.create_index('idx_epistemic_timestamp', 'epistemic_states', ['timestamp'])

    # Create behavioral_patterns table
    op.create_table('behavioral_patterns',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('pattern_id', sa.String(length=255), nullable=False),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('pattern_name', sa.String(length=255), nullable=False),
        sa.Column('execution_sequences', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('frequency', sa.Integer(), nullable=True),
        sa.Column('success_rate', sa.Float(), nullable=True),
        sa.Column('average_duration', sa.Float(), nullable=True),
        sa.Column('common_triggers', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('failure_modes', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('statistical_significance', sa.Float(), nullable=True),
        sa.Column('discovered_at', sa.DateTime(), nullable=False),
        sa.Column('last_observed', sa.DateTime(), nullable=True),
        sa.Column('pattern_type', sa.String(length=100), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_pattern_discovered', 'behavioral_patterns', ['discovered_at'])
    op.create_index('idx_pattern_success_rate', 'behavioral_patterns', ['success_rate'])
    op.create_index('idx_pattern_type', 'behavioral_patterns', ['pattern_type'])
    op.create_index(op.f('ix_behavioral_patterns_pattern_id'), 'behavioral_patterns', ['pattern_id'], unique=True)

    # Create causal_relationships table
    op.create_table('causal_relationships',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('relationship_id', sa.String(length=255), nullable=False),
        sa.Column('cause_event', sa.String(length=500), nullable=False),
        sa.Column('effect_event', sa.String(length=500), nullable=False),
        sa.Column('strength', sa.Float(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('delay_ms', sa.Integer(), nullable=True),
        sa.Column('evidence', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('statistical_significance', sa.Float(), nullable=True),
        sa.Column('causal_mechanism', sa.Text(), nullable=True),
        sa.Column('discovered_at', sa.DateTime(), nullable=False),
        sa.Column('analysis_method', sa.String(length=100), nullable=True),
        sa.Column('sample_size', sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_causal_confidence', 'causal_relationships', ['confidence'])
    op.create_index('idx_causal_discovered', 'causal_relationships', ['discovered_at'])
    op.create_index('idx_causal_method', 'causal_relationships', ['analysis_method'])
    op.create_index('idx_causal_strength', 'causal_relationships', ['strength'])
    op.create_index(op.f('ix_causal_relationships_relationship_id'), 'causal_relationships', ['relationship_id'], unique=True)

    # Create predictions table
    op.create_table('predictions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('prediction_id', sa.String(length=255), nullable=False),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('session_id', sa.String(length=255), nullable=True),
        sa.Column('prediction_type', sa.String(length=100), nullable=False),
        sa.Column('predicted_value', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('risk_factors', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('recommended_interventions', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('predicted_at', sa.DateTime(), nullable=False),
        sa.Column('target_time', sa.DateTime(), nullable=True),
        sa.Column('actual_value', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('accuracy_score', sa.Float(), nullable=True),
        sa.Column('validated_at', sa.DateTime(), nullable=True),
        sa.Column('validation_method', sa.String(length=100), nullable=True),
        sa.Column('model_name', sa.String(length=100), nullable=True),
        sa.Column('model_version', sa.String(length=50), nullable=True),
        sa.Column('feature_importance', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ),
        sa.ForeignKeyConstraint(['session_id'], ['monitoring_sessions.session_id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_prediction_accuracy', 'predictions', ['accuracy_score'])
    op.create_index('idx_prediction_agent_time', 'predictions', ['agent_id', 'predicted_at'])
    op.create_index('idx_prediction_confidence', 'predictions', ['confidence_score'])
    op.create_index('idx_prediction_time', 'predictions', ['predicted_at'])
    op.create_index('idx_prediction_type', 'predictions', ['prediction_type'])
    op.create_index(op.f('ix_predictions_prediction_id'), 'predictions', ['prediction_id'], unique=True)

    # Create system_metrics table
    op.create_table('system_metrics',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('metric_name', sa.String(length=255), nullable=False),
        sa.Column('metric_value', sa.Float(), nullable=False),
        sa.Column('metric_unit', sa.String(length=50), nullable=True),
        sa.Column('component', sa.String(length=100), nullable=True),
        sa.Column('tags', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_metrics_component', 'system_metrics', ['component'])
    op.create_index('idx_metrics_name_time', 'system_metrics', ['metric_name', 'timestamp'])
    op.create_index(op.f('ix_system_metrics_timestamp'), 'system_metrics', ['timestamp'])

    # Create audit_logs table
    op.create_table('audit_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=True),
        sa.Column('session_id', sa.String(length=255), nullable=True),
        sa.Column('operation', sa.String(length=255), nullable=False),
        sa.Column('resource_type', sa.String(length=100), nullable=True),
        sa.Column('resource_id', sa.String(length=255), nullable=True),
        sa.Column('details', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_audit_operation', 'audit_logs', ['operation'])
    op.create_index('idx_audit_resource', 'audit_logs', ['resource_type', 'resource_id'])
    op.create_index('idx_audit_timestamp', 'audit_logs', ['timestamp'])
    op.create_index('idx_audit_user', 'audit_logs', ['user_id'])


def downgrade() -> None:
    op.drop_table('audit_logs')
    op.drop_table('system_metrics')
    op.drop_table('predictions')
    op.drop_table('causal_relationships')
    op.drop_table('behavioral_patterns')
    op.drop_table('epistemic_states')
    op.drop_table('monitoring_sessions')
    op.drop_table('agents')