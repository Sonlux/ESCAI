"""Database and file management for the ESCAI framework."""

from .database import db_manager
from .base import Base
from .models import (
    Agent, MonitoringSession, EpistemicStateRecord,
    BehavioralPatternRecord, CausalRelationshipRecord,
    PredictionRecord, SystemMetrics, AuditLog
)
from .repositories import (
    AgentRepository, MonitoringSessionRepository, EpistemicStateRepository,
    BehavioralPatternRepository, CausalRelationshipRepository, PredictionRepository
)

# InfluxDB components (optional - only available if influxdb-client is installed)
try:
    from .influx_manager import InfluxDBManager, MetricPoint, RetentionPolicy
    from .influx_models import (
        MetricType, AgentPerformanceMetric, APIMetric, SystemMetric,
        PredictionMetric, PatternMetric, CausalMetric, EpistemicMetric,
        MetricBatch, create_agent_performance_metric, create_api_metric,
        create_system_metric, create_prediction_metric, create_pattern_metric,
        create_causal_metric, create_epistemic_metric
    )
    from .influx_dashboard import InfluxDashboardManager
    
    INFLUXDB_AVAILABLE = True
    
    __all__ = [
        'db_manager',
        'Base',
        'Agent',
        'MonitoringSession',
        'EpistemicStateRecord',
        'BehavioralPatternRecord',
        'CausalRelationshipRecord',
        'PredictionRecord',
        'SystemMetrics',
        'AuditLog',
        'AgentRepository',
        'MonitoringSessionRepository',
        'EpistemicStateRepository',
        'BehavioralPatternRepository',
        'CausalRelationshipRepository',
        'PredictionRepository',
        # InfluxDB components
        'InfluxDBManager',
        'MetricPoint',
        'RetentionPolicy',
        'MetricType',
        'AgentPerformanceMetric',
        'APIMetric',
        'SystemMetric',
        'PredictionMetric',
        'PatternMetric',
        'CausalMetric',
        'EpistemicMetric',
        'MetricBatch',
        'create_agent_performance_metric',
        'create_api_metric',
        'create_system_metric',
        'create_prediction_metric',
        'create_pattern_metric',
        'create_causal_metric',
        'create_epistemic_metric',
        'InfluxDashboardManager',
    ]
    
except ImportError:
    INFLUXDB_AVAILABLE = False
    
    __all__ = [
        'db_manager',
        'Base',
        'Agent',
        'MonitoringSession',
        'EpistemicStateRecord',
        'BehavioralPatternRecord',
        'CausalRelationshipRecord',
        'PredictionRecord',
        'SystemMetrics',
        'AuditLog',
        'AgentRepository',
        'MonitoringSessionRepository',
        'EpistemicStateRepository',
        'BehavioralPatternRepository',
        'CausalRelationshipRepository',
        'PredictionRepository',
    ]