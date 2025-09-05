"""
Configuration validation module for the ESCAI framework.

This module provides comprehensive validation for configuration settings,
including schema enforcement, dependency checking, and environment-specific validation.
"""

import re
import ipaddress
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from pydantic import ValidationError
import logging

from .config_schema import ConfigSchema, Environment, LogLevel


logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Configuration validation error."""
    
    def __init__(self, message: str, errors: Optional[List[Dict[str, Any]]] = None):
        super().__init__(message)
        self.errors = errors or []


class ConfigValidator:
    """Configuration validator with comprehensive validation rules."""
    
    def __init__(self):
        self.validation_rules = {
            'network': self._validate_network_settings,
            'security': self._validate_security_settings,
            'database': self._validate_database_settings,
            'environment': self._validate_environment_settings,
            'dependencies': self._validate_dependencies,
            'resources': self._validate_resource_limits
        }
    
    def validate_config(self, config_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate configuration data comprehensively.
        
        Args:
            config_data: Configuration dictionary to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        try:
            # First, validate against Pydantic schema
            config = ConfigSchema(**config_data)
            
            # Run additional validation rules
            for rule_name, rule_func in self.validation_rules.items():
                try:
                    rule_errors = rule_func(config)
                    if rule_errors:
                        errors.extend([f"{rule_name}: {error}" for error in rule_errors])
                except Exception as e:
                    errors.append(f"{rule_name} validation failed: {str(e)}")
                    
        except ValidationError as e:
            for error in e.errors():
                field = " -> ".join(str(loc) for loc in error['loc'])
                errors.append(f"Schema validation: {field}: {error['msg']}")
        except Exception as e:
            errors.append(f"Configuration validation failed: {str(e)}")
        
        return len(errors) == 0, errors
    
    def _validate_network_settings(self, config: ConfigSchema) -> List[str]:
        """Validate network-related settings."""
        errors = []
        
        # Validate API host
        try:
            if config.api.host not in ['localhost', '127.0.0.1']:
                ipaddress.ip_address(config.api.host)
        except ValueError:
            errors.append(f"Invalid API host: {config.api.host}")
        
        # Validate port ranges
        ports_to_check = [
            ('API', config.api.port),
            ('PostgreSQL', config.database.postgres_port),
            ('MongoDB', config.database.mongodb_port),
            ('Redis', config.database.redis_port),
            ('InfluxDB', config.database.influxdb_port)
        ]
        
        for service, port in ports_to_check:
            if not (1 <= port <= 65535):
                errors.append(f"Invalid {service} port: {port}")
        
        # Check for port conflicts
        used_ports = [config.api.port, config.database.postgres_port, 
                     config.database.mongodb_port, config.database.redis_port,
                     config.database.influxdb_port]
        
        if len(used_ports) != len(set(used_ports)):
            errors.append("Port conflicts detected in configuration")
        
        # Validate Neo4j URI
        if not config.database.neo4j_uri.startswith(('bolt://', 'neo4j://', 'bolt+s://', 'neo4j+s://')):
            errors.append(f"Invalid Neo4j URI format: {config.database.neo4j_uri}")
        
        return errors
    
    def _validate_security_settings(self, config: ConfigSchema) -> List[str]:
        """Validate security-related settings."""
        errors = []
        
        # Validate JWT secret key strength
        jwt_secret = config.security.jwt_secret_key.get_secret_value()
        if len(jwt_secret) < 32:
            errors.append("JWT secret key must be at least 32 characters long")
        
        # Validate TLS configuration
        if config.security.tls_enabled:
            if config.security.tls_cert_file:
                cert_path = Path(config.security.tls_cert_file)
                if not cert_path.exists():
                    errors.append(f"TLS certificate file not found: {config.security.tls_cert_file}")
            
            if config.security.tls_key_file:
                key_path = Path(config.security.tls_key_file)
                if not key_path.exists():
                    errors.append(f"TLS key file not found: {config.security.tls_key_file}")
        
        # Validate password policy consistency
        if (config.security.password_min_length < 8 and 
            config.environment == Environment.PRODUCTION):
            errors.append("Password minimum length should be at least 8 in production")
        
        # Validate session timeout
        if config.security.session_timeout_minutes < 5:
            errors.append("Session timeout should be at least 5 minutes")
        
        # Validate PII sensitivity level
        valid_pii_levels = ['low', 'medium', 'high']
        if config.security.pii_sensitivity_level not in valid_pii_levels:
            errors.append(f"Invalid PII sensitivity level: {config.security.pii_sensitivity_level}")
        
        return errors
    
    def _validate_database_settings(self, config: ConfigSchema) -> List[str]:
        """Validate database-related settings."""
        errors = []
        
        # Validate connection pool sizes
        if config.database.postgres_pool_size > config.database.postgres_max_overflow:
            errors.append("PostgreSQL max_overflow should be >= pool_size")
        
        # Validate Redis database number
        if not (0 <= config.database.redis_database <= 15):
            errors.append(f"Invalid Redis database number: {config.database.redis_database}")
        
        # Validate InfluxDB retention policy format
        retention_pattern = r'^\d+[dwmy]$'
        if not re.match(retention_pattern, config.database.influxdb_retention_policy):
            errors.append(f"Invalid InfluxDB retention policy format: {config.database.influxdb_retention_policy}")
        
        # Validate database names
        db_name_pattern = r'^[a-zA-Z][a-zA-Z0-9_]*$'
        db_names = [
            ('PostgreSQL', config.database.postgres_database),
            ('MongoDB', config.database.mongodb_database),
            ('InfluxDB', config.database.influxdb_database),
            ('Neo4j', config.database.neo4j_database)
        ]
        
        for db_type, db_name in db_names:
            if not re.match(db_name_pattern, db_name):
                errors.append(f"Invalid {db_type} database name: {db_name}")
        
        return errors
    
    def _validate_environment_settings(self, config: ConfigSchema) -> List[str]:
        """Validate environment-specific settings."""
        errors = []
        
        if config.environment == Environment.PRODUCTION:
            # Production-specific validations
            if config.debug:
                errors.append("Debug mode must be disabled in production")
            
            if config.log_level == LogLevel.DEBUG:
                errors.append("Debug logging should not be used in production")
            
            if config.api.reload:
                errors.append("API auto-reload must be disabled in production")
            
            if not config.security.tls_enabled:
                errors.append("TLS must be enabled in production")
            
            if config.monitoring.sampling_rate < 0.1:
                errors.append("Monitoring sampling rate too low for production")
        
        elif config.environment == Environment.DEVELOPMENT:
            # Development-specific validations
            if config.security.tls_enabled and not (config.security.tls_cert_file and config.security.tls_key_file):
                errors.append("TLS certificate and key files required when TLS is enabled")
        
        return errors
    
    def _validate_dependencies(self, config: ConfigSchema) -> List[str]:
        """Validate configuration dependencies."""
        errors = []
        
        # Validate monitoring dependencies
        if config.monitoring.alerting_enabled and not config.monitoring.metrics_enabled:
            errors.append("Metrics must be enabled when alerting is enabled")
        
        # Validate ML dependencies
        if config.ml.training_enabled and not config.monitoring.monitoring_enabled:
            errors.append("Monitoring must be enabled when ML training is enabled")
        
        # Validate security dependencies
        if config.security.audit_enabled and not config.monitoring.monitoring_enabled:
            errors.append("Monitoring must be enabled when audit logging is enabled")
        
        # Validate PII detection dependencies
        if config.security.pii_masking_enabled and not config.security.pii_detection_enabled:
            errors.append("PII detection must be enabled when PII masking is enabled")
        
        return errors
    
    def _validate_resource_limits(self, config: ConfigSchema) -> List[str]:
        """Validate resource limit settings."""
        errors = []
        
        # Validate API worker count
        if config.api.workers > 32:
            errors.append("API worker count should not exceed 32")
        
        # Validate connection pool sizes
        total_db_connections = (
            config.database.postgres_pool_size + 
            config.database.redis_pool_size + 
            config.database.neo4j_pool_size
        )
        
        if total_db_connections > 200:
            errors.append("Total database connection pool size exceeds recommended limit (200)")
        
        # Validate WebSocket limits
        if config.api.websocket_max_connections > 1000:
            errors.append("WebSocket connection limit exceeds recommended maximum (1000)")
        
        # Validate ML model cache
        if config.ml.model_cache_size > 1000:
            errors.append("ML model cache size exceeds recommended limit (1000)")
        
        # Validate monitoring overhead
        if config.monitoring.monitoring_overhead_threshold > 0.2:
            errors.append("Monitoring overhead threshold too high (>20%)")
        
        return errors
    
    def validate_config_file(self, config_file_path: str) -> Tuple[bool, List[str]]:
        """
        Validate configuration from file.
        
        Args:
            config_file_path: Path to configuration file
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        try:
            import yaml
            import json
            
            config_path = Path(config_file_path)
            if not config_path.exists():
                return False, [f"Configuration file not found: {config_file_path}"]
            
            # Load configuration based on file extension
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yml', '.yaml']:
                    config_data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    return False, [f"Unsupported configuration file format: {config_path.suffix}"]
            
            if not isinstance(config_data, dict):
                return False, [f"Invalid configuration file format: Expected a dictionary, got {type(config_data).__name__}"]
            
            return self.validate_config(config_data)
            
        except Exception as e:
            return False, [f"Failed to load configuration file: {str(e)}"]
    
    def generate_validation_report(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Args:
            config_data: Configuration dictionary to validate
            
        Returns:
            Validation report dictionary
        """
        is_valid, errors = self.validate_config(config_data)
        
        from datetime import datetime
        
        report: Dict[str, Any] = {
            'valid': is_valid,
            'timestamp': datetime.now().isoformat(),
            'total_errors': len(errors),
            'errors': errors,
            'warnings': [],
            'recommendations': []
        }
        
        # Add warnings and recommendations based on configuration
        try:
            config = ConfigSchema(**config_data)
            
            # Performance warnings
            if config.monitoring.sampling_rate == 1.0:
                report['warnings'].append("100% sampling rate may impact performance")
            
            if config.api.workers == 1:
                report['warnings'].append("Single API worker may limit throughput")
            
            # Security recommendations
            if config.environment == Environment.PRODUCTION:
                if config.security.jwt_access_token_expire_minutes > 60:
                    report['recommendations'].append("Consider shorter JWT token expiry in production")
                
                if not config.security.audit_enabled:
                    report['recommendations'].append("Enable audit logging in production")
            
            # Performance recommendations
            if config.database.postgres_pool_size < 5:
                report['recommendations'].append("Consider increasing PostgreSQL connection pool size")
            
        except Exception as e:
            report['warnings'].append(f"Could not generate recommendations: {str(e)}")
        
        return report