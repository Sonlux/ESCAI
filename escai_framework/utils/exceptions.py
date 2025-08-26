"""
Centralized exception hierarchy for the ESCAI framework.

This module defines all custom exceptions used throughout the framework,
organized in a hierarchical structure for proper error handling and recovery.
"""

from typing import Any, Dict, Optional, List
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels for categorizing exceptions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors for classification and handling."""
    INSTRUMENTATION = "instrumentation"
    PROCESSING = "processing"
    STORAGE = "storage"
    API = "api"
    NETWORK = "network"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    CONFIGURATION = "configuration"


class ESCAIBaseException(Exception):
    """
    Base exception class for all ESCAI framework exceptions.
    
    Provides common functionality for error tracking, context, and recovery hints.
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = None,
        context: Dict[str, Any] = None,
        recovery_hint: str = None,
        cause: Exception = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.severity = severity
        self.category = category
        self.context = context or {}
        self.recovery_hint = recovery_hint
        self.cause = cause
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging and API responses."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value if self.category else None,
            "context": self.context,
            "recovery_hint": self.recovery_hint,
            "cause": str(self.cause) if self.cause else None
        }


# Instrumentation Exceptions
class InstrumentationError(ESCAIBaseException):
    """Base class for instrumentation-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.INSTRUMENTATION,
            **kwargs
        )


class FrameworkNotSupportedError(InstrumentationError):
    """Raised when trying to instrument an unsupported framework."""
    
    def __init__(self, framework_name: str, supported_frameworks: List[str]):
        super().__init__(
            f"Framework '{framework_name}' is not supported. "
            f"Supported frameworks: {', '.join(supported_frameworks)}",
            severity=ErrorSeverity.HIGH,
            context={"framework": framework_name, "supported": supported_frameworks},
            recovery_hint="Use one of the supported frameworks or implement a custom instrumentor"
        )


class MonitoringOverheadError(InstrumentationError):
    """Raised when monitoring overhead exceeds acceptable thresholds."""
    
    def __init__(self, current_overhead: float, threshold: float):
        super().__init__(
            f"Monitoring overhead ({current_overhead:.2%}) exceeds threshold ({threshold:.2%})",
            severity=ErrorSeverity.HIGH,
            context={"current_overhead": current_overhead, "threshold": threshold},
            recovery_hint="Reduce monitoring frequency or disable non-critical features"
        )


class AgentConnectionError(InstrumentationError):
    """Raised when unable to connect to or monitor an agent."""
    
    def __init__(self, agent_id: str, reason: str):
        super().__init__(
            f"Failed to connect to agent '{agent_id}': {reason}",
            severity=ErrorSeverity.MEDIUM,
            context={"agent_id": agent_id, "reason": reason},
            recovery_hint="Check agent configuration and network connectivity"
        )


# Processing Exceptions
class ProcessingError(ESCAIBaseException):
    """Base class for data processing errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.PROCESSING,
            **kwargs
        )


class EpistemicExtractionError(ProcessingError):
    """Raised when epistemic state extraction fails."""
    
    def __init__(self, reason: str, data_sample: str = None):
        super().__init__(
            f"Failed to extract epistemic state: {reason}",
            severity=ErrorSeverity.MEDIUM,
            context={"reason": reason, "data_sample": data_sample[:100] if data_sample else None},
            recovery_hint="Check input data format or use fallback extraction method"
        )


class PatternAnalysisError(ProcessingError):
    """Raised when behavioral pattern analysis fails."""
    
    def __init__(self, reason: str, pattern_count: int = None):
        super().__init__(
            f"Pattern analysis failed: {reason}",
            severity=ErrorSeverity.MEDIUM,
            context={"reason": reason, "pattern_count": pattern_count},
            recovery_hint="Reduce pattern complexity or increase data sample size"
        )


class CausalAnalysisError(ProcessingError):
    """Raised when causal inference fails."""
    
    def __init__(self, reason: str, event_count: int = None):
        super().__init__(
            f"Causal analysis failed: {reason}",
            severity=ErrorSeverity.MEDIUM,
            context={"reason": reason, "event_count": event_count},
            recovery_hint="Ensure sufficient temporal data or adjust causality parameters"
        )


class PredictionError(ProcessingError):
    """Raised when performance prediction fails."""
    
    def __init__(self, model_name: str, reason: str):
        super().__init__(
            f"Prediction failed with model '{model_name}': {reason}",
            severity=ErrorSeverity.MEDIUM,
            context={"model_name": model_name, "reason": reason},
            recovery_hint="Try alternative prediction model or retrain with more data"
        )


class ModelLoadError(ProcessingError):
    """Raised when ML model loading fails."""
    
    def __init__(self, model_path: str, reason: str):
        super().__init__(
            f"Failed to load model from '{model_path}': {reason}",
            severity=ErrorSeverity.HIGH,
            context={"model_path": model_path, "reason": reason},
            recovery_hint="Check model file exists and is compatible with current version"
        )


# Storage Exceptions
class StorageError(ESCAIBaseException):
    """Base class for storage-related errors."""
    
    def __init__(self, message: str, error_code: str = None, **kwargs):
        super().__init__(
            message,
            error_code=error_code,
            category=ErrorCategory.STORAGE,
            **kwargs
        )


class DatabaseConnectionError(StorageError):
    """Raised when database connection fails."""
    
    def __init__(self, database_type: str, connection_string: str, reason: str):
        super().__init__(
            f"Failed to connect to {database_type} database: {reason}",
            severity=ErrorSeverity.HIGH,
            context={
                "database_type": database_type,
                "connection_string": connection_string,
                "reason": reason
            },
            recovery_hint="Check database configuration and network connectivity"
        )


class DataValidationError(StorageError):
    """Raised when data validation fails before storage."""
    
    def __init__(self, field_name: str, value: Any, expected_type: str):
        super().__init__(
            f"Validation failed for field '{field_name}': expected {expected_type}, got {type(value).__name__}",
            severity=ErrorSeverity.MEDIUM,
            context={"field": field_name, "value": str(value), "expected_type": expected_type},
            recovery_hint="Check data format and ensure all required fields are present"
        )


class StorageCapacityError(StorageError):
    """Raised when storage capacity limits are reached."""
    
    def __init__(self, storage_type: str, current_usage: float, limit: float):
        super().__init__(
            f"{storage_type} storage capacity exceeded: {current_usage:.1f}GB / {limit:.1f}GB",
            severity=ErrorSeverity.CRITICAL,
            context={"storage_type": storage_type, "usage": current_usage, "limit": limit},
            recovery_hint="Archive old data or increase storage capacity"
        )


# API Exceptions
class APIError(ESCAIBaseException):
    """Base class for API-related errors."""
    
    def __init__(self, message: str, status_code: int = 500, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.API,
            **kwargs
        )
        self.status_code = status_code


class AuthenticationError(APIError):
    """Raised when authentication fails."""
    
    def __init__(self, reason: str):
        super().__init__(
            f"Authentication failed: {reason}",
            status_code=401,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.AUTHENTICATION,
            recovery_hint="Check credentials or refresh authentication token"
        )


class AuthorizationError(APIError):
    """Raised when authorization fails."""
    
    def __init__(self, user_id: str, required_permission: str):
        super().__init__(
            f"User '{user_id}' lacks required permission: {required_permission}",
            status_code=403,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.AUTHENTICATION,
            context={"user_id": user_id, "permission": required_permission},
            recovery_hint="Contact administrator to request required permissions"
        )


class RateLimitError(APIError):
    """Raised when API rate limits are exceeded."""
    
    def __init__(self, limit: int, window_seconds: int, retry_after: int):
        super().__init__(
            f"Rate limit exceeded: {limit} requests per {window_seconds} seconds",
            status_code=429,
            severity=ErrorSeverity.LOW,
            context={"limit": limit, "window": window_seconds, "retry_after": retry_after},
            recovery_hint=f"Wait {retry_after} seconds before retrying"
        )


class ValidationError(APIError):
    """Raised when request validation fails."""
    
    def __init__(self, field_errors: Dict[str, str]):
        super().__init__(
            f"Validation failed: {', '.join(f'{k}: {v}' for k, v in field_errors.items())}",
            status_code=400,
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.VALIDATION,
            context={"field_errors": field_errors},
            recovery_hint="Check request format and ensure all required fields are provided"
        )


# Network Exceptions
class NetworkError(ESCAIBaseException):
    """Base class for network-related errors."""
    
    def __init__(self, message: str, error_code: str = None, **kwargs):
        super().__init__(
            message,
            error_code=error_code,
            category=ErrorCategory.NETWORK,
            **kwargs
        )


class ConnectionTimeoutError(NetworkError):
    """Raised when network connections timeout."""
    
    def __init__(self, service: str, timeout_seconds: float):
        super().__init__(
            f"Connection to {service} timed out after {timeout_seconds} seconds",
            severity=ErrorSeverity.MEDIUM,
            context={"service": service, "timeout": timeout_seconds},
            recovery_hint="Check network connectivity or increase timeout value"
        )


class ServiceUnavailableError(NetworkError):
    """Raised when external services are unavailable."""
    
    def __init__(self, service: str, status_code: int = None):
        super().__init__(
            f"Service '{service}' is unavailable" + (f" (HTTP {status_code})" if status_code else ""),
            severity=ErrorSeverity.HIGH,
            context={"service": service, "status_code": status_code},
            recovery_hint="Wait for service to recover or use fallback mechanism"
        )


# Configuration Exceptions
class ConfigurationError(ESCAIBaseException):
    """Base class for configuration-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            **kwargs
        )


class MissingConfigurationError(ConfigurationError):
    """Raised when required configuration is missing."""
    
    def __init__(self, config_key: str, config_file: str = None):
        super().__init__(
            f"Missing required configuration: {config_key}" + (f" in {config_file}" if config_file else ""),
            severity=ErrorSeverity.HIGH,
            context={"config_key": config_key, "config_file": config_file},
            recovery_hint="Add the required configuration value or use environment variable"
        )


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration values are invalid."""
    
    def __init__(self, config_key: str, value: Any, expected: str):
        super().__init__(
            f"Invalid configuration for '{config_key}': expected {expected}, got {value}",
            severity=ErrorSeverity.HIGH,
            context={"config_key": config_key, "value": str(value), "expected": expected},
            recovery_hint="Check configuration format and valid value ranges"
        )