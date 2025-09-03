"""
Configuration schema definitions for the ESCAI framework.

This module defines Pydantic models for configuration validation and type safety.
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator, SecretStr
from enum import Enum


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Environment(str, Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class DatabaseConfig(BaseModel):
    """Database configuration schema."""
    
    # PostgreSQL Configuration
    postgres_host: str = Field(default="localhost", description="PostgreSQL host")
    postgres_port: int = Field(default=5432, ge=1, le=65535, description="PostgreSQL port")
    postgres_database: str = Field(default="escai", description="PostgreSQL database name")
    postgres_username: str = Field(default="escai_user", description="PostgreSQL username")
    postgres_password: SecretStr = Field(default=SecretStr(""), description="PostgreSQL password")
    postgres_pool_size: int = Field(default=10, ge=1, le=100, description="Connection pool size")
    postgres_max_overflow: int = Field(default=20, ge=0, le=100, description="Max pool overflow")
    
    # MongoDB Configuration
    mongodb_host: str = Field(default="localhost", description="MongoDB host")
    mongodb_port: int = Field(default=27017, ge=1, le=65535, description="MongoDB port")
    mongodb_database: str = Field(default="escai_logs", description="MongoDB database name")
    mongodb_username: Optional[str] = Field(default=None, description="MongoDB username")
    mongodb_password: Optional[SecretStr] = Field(default=None, description="MongoDB password")
    mongodb_replica_set: Optional[str] = Field(default=None, description="MongoDB replica set")
    
    # Redis Configuration
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, ge=1, le=65535, description="Redis port")
    redis_database: int = Field(default=0, ge=0, le=15, description="Redis database number")
    redis_password: Optional[SecretStr] = Field(default=None, description="Redis password")
    redis_pool_size: int = Field(default=10, ge=1, le=100, description="Redis connection pool size")
    
    # InfluxDB Configuration
    influxdb_host: str = Field(default="localhost", description="InfluxDB host")
    influxdb_port: int = Field(default=8086, ge=1, le=65535, description="InfluxDB port")
    influxdb_database: str = Field(default="escai_metrics", description="InfluxDB database name")
    influxdb_username: Optional[str] = Field(default=None, description="InfluxDB username")
    influxdb_password: Optional[SecretStr] = Field(default=None, description="InfluxDB password")
    influxdb_retention_policy: str = Field(default="30d", description="Data retention policy")
    
    # Neo4j Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    neo4j_username: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: SecretStr = Field(default=SecretStr(""), description="Neo4j password")
    neo4j_database: str = Field(default="neo4j", description="Neo4j database name")
    neo4j_pool_size: int = Field(default=10, ge=1, le=100, description="Neo4j connection pool size")


class APIConfig(BaseModel):
    """API configuration schema."""
    
    host: str = Field(default="127.0.0.1", description="API host address")
    port: int = Field(default=8000, ge=1, le=65535, description="API port")
    workers: int = Field(default=4, ge=1, le=32, description="Number of worker processes")
    reload: bool = Field(default=False, description="Enable auto-reload in development")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, ge=1, description="Requests per minute per IP")
    rate_limit_window: int = Field(default=60, ge=1, description="Rate limit window in seconds")
    
    # CORS settings
    cors_origins: List[str] = Field(default=["*"], description="Allowed CORS origins")
    cors_methods: List[str] = Field(default=["GET", "POST", "PUT", "DELETE"], description="Allowed CORS methods")
    cors_headers: List[str] = Field(default=["*"], description="Allowed CORS headers")
    
    # Request/Response limits
    max_request_size: int = Field(default=10485760, ge=1024, description="Max request size in bytes (10MB)")
    request_timeout: int = Field(default=30, ge=1, description="Request timeout in seconds")
    
    # WebSocket settings
    websocket_max_connections: int = Field(default=100, ge=1, description="Max WebSocket connections")
    websocket_heartbeat_interval: int = Field(default=30, ge=5, description="WebSocket heartbeat interval")


class SecurityConfig(BaseModel):
    """Security configuration schema."""
    
    # JWT Configuration
    jwt_secret_key: SecretStr = Field(description="JWT secret key for token signing")
    jwt_algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    jwt_access_token_expire_minutes: int = Field(default=30, ge=1, description="Access token expiry")
    jwt_refresh_token_expire_days: int = Field(default=7, ge=1, description="Refresh token expiry")
    
    # TLS Configuration
    tls_enabled: bool = Field(default=True, description="Enable TLS encryption")
    tls_cert_file: Optional[str] = Field(default=None, description="TLS certificate file path")
    tls_key_file: Optional[str] = Field(default=None, description="TLS private key file path")
    tls_ca_file: Optional[str] = Field(default=None, description="TLS CA certificate file path")
    
    # Password policies
    password_min_length: int = Field(default=8, ge=6, description="Minimum password length")
    password_require_uppercase: bool = Field(default=True, description="Require uppercase letters")
    password_require_lowercase: bool = Field(default=True, description="Require lowercase letters")
    password_require_numbers: bool = Field(default=True, description="Require numbers")
    password_require_symbols: bool = Field(default=True, description="Require symbols")
    
    # Session management
    session_timeout_minutes: int = Field(default=60, ge=5, description="Session timeout")
    max_login_attempts: int = Field(default=5, ge=1, description="Max failed login attempts")
    lockout_duration_minutes: int = Field(default=15, ge=1, description="Account lockout duration")
    
    # PII Detection
    pii_detection_enabled: bool = Field(default=True, description="Enable PII detection")
    pii_masking_enabled: bool = Field(default=True, description="Enable PII masking")
    pii_sensitivity_level: str = Field(default="medium", description="PII detection sensitivity")
    
    # Audit logging
    audit_enabled: bool = Field(default=True, description="Enable audit logging")
    audit_retention_days: int = Field(default=90, ge=1, description="Audit log retention period")


class MonitoringConfig(BaseModel):
    """Monitoring and observability configuration."""
    
    # Performance monitoring
    monitoring_enabled: bool = Field(default=True, description="Enable performance monitoring")
    monitoring_overhead_threshold: float = Field(default=0.1, ge=0.01, le=0.5, description="Max monitoring overhead")
    sampling_rate: float = Field(default=1.0, ge=0.01, le=1.0, description="Event sampling rate")
    
    # Metrics collection
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    metrics_interval_seconds: int = Field(default=60, ge=1, description="Metrics collection interval")
    
    # Health checks
    health_check_enabled: bool = Field(default=True, description="Enable health checks")
    health_check_interval_seconds: int = Field(default=30, ge=5, description="Health check interval")
    
    # Alerting
    alerting_enabled: bool = Field(default=True, description="Enable alerting")
    alert_thresholds: Dict[str, float] = Field(
        default={
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "error_rate": 5.0
        },
        description="Alert thresholds for various metrics"
    )


class MLConfig(BaseModel):
    """Machine learning configuration."""
    
    # Model settings
    model_cache_enabled: bool = Field(default=True, description="Enable model caching")
    model_cache_size: int = Field(default=100, ge=1, description="Model cache size")
    model_update_interval_hours: int = Field(default=24, ge=1, description="Model update interval")
    
    # Training settings
    training_enabled: bool = Field(default=True, description="Enable model training")
    training_batch_size: int = Field(default=32, ge=1, description="Training batch size")
    training_epochs: int = Field(default=10, ge=1, description="Training epochs")
    validation_split: float = Field(default=0.2, ge=0.1, le=0.5, description="Validation data split")
    
    # Prediction settings
    prediction_confidence_threshold: float = Field(default=0.7, ge=0.1, le=1.0, description="Prediction confidence threshold")
    ensemble_size: int = Field(default=5, ge=1, le=20, description="Ensemble model size")


class ConfigSchema(BaseModel):
    """Main configuration schema for the ESCAI framework."""
    
    # Environment settings
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Deployment environment")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    
    # Component configurations
    database: DatabaseConfig = Field(default_factory=lambda: DatabaseConfig(), description="Database configuration")
    api: APIConfig = Field(default_factory=lambda: APIConfig(), description="API configuration")
    security: SecurityConfig = Field(description="Security configuration")
    monitoring: MonitoringConfig = Field(default_factory=lambda: MonitoringConfig(), description="Monitoring configuration")
    ml: MLConfig = Field(default_factory=lambda: MLConfig(), description="Machine learning configuration")
    
    # Custom settings
    custom_settings: Dict[str, Any] = Field(default_factory=dict, description="Custom application settings")
    
    @validator('environment')
    def validate_environment(cls, v):
        """Validate environment setting."""
        if v == Environment.PRODUCTION:
            # Additional validation for production
            pass
        return v
    
    @validator('debug')
    def validate_debug_in_production(cls, v, values):
        """Ensure debug is disabled in production."""
        if values.get('environment') == Environment.PRODUCTION and v:
            raise ValueError("Debug mode must be disabled in production")
        return v
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
        extra = "forbid"