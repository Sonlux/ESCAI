"""
ESCAI Framework Security Module

This module provides comprehensive security features including:
- TLS 1.3 encryption and certificate management
- Enhanced JWT authentication with refresh tokens
- Role-based access control (RBAC)
- PII detection and masking
- Audit logging with tamper-proof storage
- Input validation and sanitization
"""

from .tls_manager import TLSManager
from .auth_manager import AuthManager, TokenManager
from .rbac import RBACManager, Permission, Role, ResourceType, Action
from .pii_detector import PIIDetector, PIIMasker, PIIType, SensitivityLevel
from .audit_logger import AuditLogger, AuditEvent, AuditEventType, AuditLevel
from .input_validator import InputValidator, SecuritySanitizer, ValidationLevel

__all__ = [
    'TLSManager',
    'AuthManager',
    'TokenManager', 
    'RBACManager',
    'Permission',
    'Role',
    'ResourceType',
    'Action',
    'PIIDetector',
    'PIIMasker',
    'PIIType',
    'SensitivityLevel',
    'AuditLogger',
    'AuditEvent',
    'AuditEventType',
    'AuditLevel',
    'InputValidator',
    'SecuritySanitizer',
    'ValidationLevel'
]