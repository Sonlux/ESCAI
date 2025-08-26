"""
Utility modules for the ESCAI framework.
"""

from .validation import ValidationError, validate_string, validate_number, validate_probability
from .serialization import to_json, from_json, to_dict, from_dict

# Error handling and resilience utilities
from .exceptions import (
    ESCAIBaseException, InstrumentationError, ProcessingError,
    StorageError, APIError, NetworkError, ConfigurationError,
    ErrorSeverity, ErrorCategory
)
from .retry import (
    RetryManager, RetryConfig, BackoffStrategy,
    retry_async, retry_sync, retry_database, retry_api, retry_ml_model
)
from .circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitState,
    MonitoringCircuitBreaker, get_circuit_breaker, get_monitoring_circuit_breaker
)
from .fallback import (
    FallbackManager, FallbackResult, FallbackStrategy,
    get_fallback_manager, execute_with_fallback
)
from .load_shedding import (
    LoadShedder, LoadLevel, Priority, GracefulDegradationManager,
    get_degradation_manager, execute_with_degradation
)
from .error_tracking import (
    ErrorTracker, ErrorEvent, StructuredLogger,
    get_error_tracker, get_logger, monitor_errors, track_exception
)

__all__ = [
    # Validation and serialization
    'ValidationError',
    'validate_string',
    'validate_number',
    'validate_probability',
    'to_json',
    'from_json',
    'to_dict',
    'from_dict',
    
    # Exception hierarchy
    'ESCAIBaseException',
    'InstrumentationError',
    'ProcessingError',
    'StorageError',
    'APIError',
    'NetworkError',
    'ConfigurationError',
    'ErrorSeverity',
    'ErrorCategory',
    
    # Retry mechanisms
    'RetryManager',
    'RetryConfig',
    'BackoffStrategy',
    'retry_async',
    'retry_sync',
    'retry_database',
    'retry_api',
    'retry_ml_model',
    
    # Circuit breakers
    'CircuitBreaker',
    'CircuitBreakerConfig',
    'CircuitState',
    'MonitoringCircuitBreaker',
    'get_circuit_breaker',
    'get_monitoring_circuit_breaker',
    
    # Fallback mechanisms
    'FallbackManager',
    'FallbackResult',
    'FallbackStrategy',
    'get_fallback_manager',
    'execute_with_fallback',
    
    # Load shedding and degradation
    'LoadShedder',
    'LoadLevel',
    'Priority',
    'GracefulDegradationManager',
    'get_degradation_manager',
    'execute_with_degradation',
    
    # Error tracking
    'ErrorTracker',
    'ErrorEvent',
    'StructuredLogger',
    'get_error_tracker',
    'get_logger',
    'monitor_errors',
    'track_exception'
]