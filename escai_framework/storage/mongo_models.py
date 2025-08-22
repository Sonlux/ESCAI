"""
MongoDB document models for unstructured data in ESCAI Framework.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, validator
from bson import ObjectId


class PyObjectId(ObjectId):
    """Custom ObjectId type for Pydantic models."""
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema
        return core_schema.no_info_plain_validator_function(cls.validate)
    
    @classmethod
    def validate(cls, v):
        if isinstance(v, ObjectId):
            return v
        if isinstance(v, str) and ObjectId.is_valid(v):
            return ObjectId(v)
        raise ValueError("Invalid ObjectId")
    
    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema, handler):
        field_schema.update(type="string", format="objectid")
        return field_schema


class MongoBaseModel(BaseModel):
    """Base model for MongoDB documents."""
    
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
    
    def dict(self, **kwargs):
        """Override dict to handle ObjectId serialization."""
        d = super().dict(**kwargs)
        if "_id" in d:
            d["_id"] = str(d["_id"])
        return d


class RawLogDocument(MongoBaseModel):
    """Document model for raw agent execution logs."""
    
    agent_id: str = Field(..., description="Unique identifier for the agent")
    session_id: str = Field(..., description="Monitoring session identifier")
    framework: str = Field(..., description="Agent framework (langchain, autogen, etc.)")
    log_level: str = Field(..., description="Log level (DEBUG, INFO, WARNING, ERROR)")
    message: str = Field(..., description="Raw log message")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime = Field(..., description="Log timestamp")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}")
        return v.upper()
    
    @validator('framework')
    def validate_framework(cls, v):
        valid_frameworks = ['langchain', 'autogen', 'crewai', 'openai', 'custom']
        if v.lower() not in valid_frameworks:
            raise ValueError(f"Invalid framework: {v}")
        return v.lower()


class ProcessedEventDocument(MongoBaseModel):
    """Document model for processed agent events."""
    
    agent_id: str = Field(..., description="Unique identifier for the agent")
    session_id: str = Field(..., description="Monitoring session identifier")
    event_type: str = Field(..., description="Type of event")
    event_data: Dict[str, Any] = Field(..., description="Event payload")
    processing_metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata")
    source_log_ids: List[str] = Field(default_factory=list, description="Source log document IDs")
    timestamp: datetime = Field(..., description="Event timestamp")
    processed_at: datetime = Field(default_factory=datetime.utcnow, description="Processing timestamp")
    
    @validator('event_type')
    def validate_event_type(cls, v):
        valid_types = [
            'agent_start', 'agent_stop', 'decision_made', 'tool_used',
            'memory_updated', 'goal_changed', 'belief_updated', 'error_occurred',
            'pattern_detected', 'anomaly_detected'
        ]
        if v not in valid_types:
            raise ValueError(f"Invalid event type: {v}")
        return v


class ExplanationDocument(MongoBaseModel):
    """Document model for generated explanations."""
    
    agent_id: str = Field(..., description="Unique identifier for the agent")
    session_id: str = Field(..., description="Monitoring session identifier")
    explanation_type: str = Field(..., description="Type of explanation")
    title: str = Field(..., description="Explanation title")
    content: str = Field(..., description="Human-readable explanation content")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in explanation")
    supporting_evidence: List[Dict[str, Any]] = Field(default_factory=list, description="Supporting evidence")
    related_events: List[str] = Field(default_factory=list, description="Related event IDs")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('explanation_type')
    def validate_explanation_type(cls, v):
        valid_types = [
            'behavior_summary', 'decision_pathway', 'causal_explanation',
            'failure_analysis', 'success_factors', 'pattern_explanation'
        ]
        if v not in valid_types:
            raise ValueError(f"Invalid explanation type: {v}")
        return v


class ConfigurationDocument(MongoBaseModel):
    """Document model for system and user configurations."""
    
    config_type: str = Field(..., description="Type of configuration")
    config_name: str = Field(..., description="Configuration name")
    config_data: Dict[str, Any] = Field(..., description="Configuration data")
    user_id: Optional[str] = Field(None, description="User ID for user-specific configs")
    agent_id: Optional[str] = Field(None, description="Agent ID for agent-specific configs")
    version: int = Field(default=1, description="Configuration version")
    is_active: bool = Field(default=True, description="Whether configuration is active")
    
    @validator('config_type')
    def validate_config_type(cls, v):
        valid_types = [
            'system', 'user_preferences', 'agent_settings', 'monitoring_rules',
            'alert_rules', 'dashboard_layout', 'api_settings'
        ]
        if v not in valid_types:
            raise ValueError(f"Invalid configuration type: {v}")
        return v


class AnalyticsResultDocument(MongoBaseModel):
    """Document model for analytics and ML model results."""
    
    analysis_type: str = Field(..., description="Type of analysis performed")
    agent_id: Optional[str] = Field(None, description="Agent ID if analysis is agent-specific")
    session_id: Optional[str] = Field(None, description="Session ID if analysis is session-specific")
    model_name: str = Field(..., description="Name of the model or algorithm used")
    model_version: str = Field(..., description="Version of the model")
    input_data_hash: str = Field(..., description="Hash of input data for reproducibility")
    results: Dict[str, Any] = Field(..., description="Analysis results")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Performance metrics")
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")
    
    @validator('analysis_type')
    def validate_analysis_type(cls, v):
        valid_types = [
            'pattern_mining', 'anomaly_detection', 'causal_inference',
            'performance_prediction', 'behavioral_clustering', 'failure_analysis'
        ]
        if v not in valid_types:
            raise ValueError(f"Invalid analysis type: {v}")
        return v