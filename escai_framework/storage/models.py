"""
SQLAlchemy models for structured data storage in PostgreSQL.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
import json

from sqlalchemy import (
    Column, Integer, String, DateTime, Float, Text, Boolean,
    ForeignKey, JSON, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid

from .base import Base


class Agent(Base):
    """Agent registry and metadata."""
    
    __tablename__ = 'agents'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    framework = Column(String(100), nullable=False)  # langchain, autogen, crewai, openai
    version = Column(String(50))
    description = Column(Text)
    configuration = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True, nullable=False, server_default='true')
    
    # Relationships
    epistemic_states = relationship("EpistemicStateRecord", back_populates="agent", cascade="all, delete-orphan")
    behavioral_patterns = relationship("BehavioralPatternRecord", back_populates="agent", cascade="all, delete-orphan")
    predictions = relationship("PredictionRecord", back_populates="agent", cascade="all, delete-orphan")
    monitoring_sessions = relationship("MonitoringSession", back_populates="agent", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_agent_framework', 'framework'),
        Index('idx_agent_active', 'is_active'),
    )


class MonitoringSession(Base):
    """Monitoring session tracking and configuration."""
    
    __tablename__ = 'monitoring_sessions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(255), unique=True, nullable=False, index=True)
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agents.id'), nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ended_at = Column(DateTime)
    status = Column(String(50), default='active', nullable=False)  # active, completed, failed
    configuration = Column(JSONB)
    session_metadata = Column(JSONB)
    
    # Relationships
    agent = relationship("Agent", back_populates="monitoring_sessions")
    
    __table_args__ = (
        Index('idx_session_status', 'status'),
        Index('idx_session_started', 'started_at'),
    )


class EpistemicStateRecord(Base):
    """Time-series epistemic state data."""
    
    __tablename__ = 'epistemic_states'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agents.id'), nullable=False)
    session_id = Column(String(255), ForeignKey('monitoring_sessions.session_id'))
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Epistemic state components
    belief_states = Column(JSONB)
    knowledge_state = Column(JSONB)
    goal_state = Column(JSONB)
    confidence_level = Column(Float)
    uncertainty_score = Column(Float)
    decision_context = Column(JSONB)
    
    # Metadata
    extraction_method = Column(String(100))
    processing_time_ms = Column(Integer)
    
    # Relationships
    agent = relationship("Agent", back_populates="epistemic_states")
    
    __table_args__ = (
        Index('idx_epistemic_timestamp', 'timestamp'),
        Index('idx_epistemic_agent_time', 'agent_id', 'timestamp'),
        Index('idx_epistemic_session', 'session_id'),
    )


class BehavioralPatternRecord(Base):
    """Identified behavioral patterns and metadata."""
    
    __tablename__ = 'behavioral_patterns'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pattern_id = Column(String(255), unique=True, nullable=False, index=True)
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agents.id'), nullable=False)
    pattern_name = Column(String(255), nullable=False)
    
    # Pattern characteristics
    execution_sequences = Column(JSONB)
    frequency = Column(Integer, default=0)
    success_rate = Column(Float)
    average_duration = Column(Float)
    common_triggers = Column(JSONB)
    failure_modes = Column(JSONB)
    statistical_significance = Column(Float)
    
    # Metadata
    discovered_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_observed = Column(DateTime)
    pattern_type = Column(String(100))  # sequential, temporal, causal
    confidence_score = Column(Float)
    
    # Relationships
    agent = relationship("Agent", back_populates="behavioral_patterns")
    
    __table_args__ = (
        Index('idx_pattern_type', 'pattern_type'),
        Index('idx_pattern_discovered', 'discovered_at'),
        Index('idx_pattern_success_rate', 'success_rate'),
    )


class CausalRelationshipRecord(Base):
    """Discovered causal relationships."""
    
    __tablename__ = 'causal_relationships'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    relationship_id = Column(String(255), unique=True, nullable=False, index=True)
    
    # Causal relationship components
    cause_event = Column(String(500), nullable=False)
    effect_event = Column(String(500), nullable=False)
    strength = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    delay_ms = Column(Integer)
    evidence = Column(JSONB)
    statistical_significance = Column(Float)
    causal_mechanism = Column(Text)
    
    # Metadata
    discovered_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    analysis_method = Column(String(100))  # granger, structural, temporal
    sample_size = Column(Integer)
    
    __table_args__ = (
        Index('idx_causal_strength', 'strength'),
        Index('idx_causal_confidence', 'confidence'),
        Index('idx_causal_method', 'analysis_method'),
        Index('idx_causal_discovered', 'discovered_at'),
    )


class PredictionRecord(Base):
    """Prediction results and accuracy tracking."""
    
    __tablename__ = 'predictions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prediction_id = Column(String(255), unique=True, nullable=False, index=True)
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agents.id'), nullable=False)
    session_id = Column(String(255), ForeignKey('monitoring_sessions.session_id'))
    
    # Prediction details
    prediction_type = Column(String(100), nullable=False)  # success, failure, completion_time
    predicted_value = Column(JSONB)
    confidence_score = Column(Float)
    risk_factors = Column(JSONB)
    recommended_interventions = Column(JSONB)
    
    # Timing
    predicted_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    target_time = Column(DateTime)  # When prediction is for
    
    # Validation
    actual_value = Column(JSONB)
    accuracy_score = Column(Float)
    validated_at = Column(DateTime)
    validation_method = Column(String(100))
    
    # Model information
    model_name = Column(String(100))
    model_version = Column(String(50))
    feature_importance = Column(JSONB)
    
    # Relationships
    agent = relationship("Agent", back_populates="predictions")
    
    __table_args__ = (
        Index('idx_prediction_type', 'prediction_type'),
        Index('idx_prediction_time', 'predicted_at'),
        Index('idx_prediction_confidence', 'confidence_score'),
        Index('idx_prediction_accuracy', 'accuracy_score'),
        Index('idx_prediction_agent_time', 'agent_id', 'predicted_at'),
    )


class SystemMetrics(Base):
    """System performance and health metrics."""
    
    __tablename__ = 'system_metrics'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    metric_name = Column(String(255), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(50))
    component = Column(String(100))  # api, processor, storage, etc.
    tags = Column(JSONB)
    
    __table_args__ = (
        Index('idx_metrics_name_time', 'metric_name', 'timestamp'),
        Index('idx_metrics_component', 'component'),
    )


class AuditLog(Base):
    """Audit logging for all operations."""
    
    __tablename__ = 'audit_logs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    user_id = Column(String(255))
    session_id = Column(String(255))
    operation = Column(String(255), nullable=False)
    resource_type = Column(String(100))
    resource_id = Column(String(255))
    details = Column(JSONB)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    
    __table_args__ = (
        Index('idx_audit_operation', 'operation'),
        Index('idx_audit_resource', 'resource_type', 'resource_id'),
        Index('idx_audit_user', 'user_id'),
        Index('idx_audit_timestamp', 'timestamp'),
    )