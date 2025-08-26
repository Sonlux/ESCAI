"""
Security Configuration Management

Centralized configuration for all security components with
environment-specific settings and secure defaults.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import secrets


class SecurityProfile(Enum):
    """Security configuration profiles"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"
    HIGH_SECURITY = "high_security"


@dataclass
class TLSConfig:
    """TLS configuration settings"""
    cert_dir: str = "certs"
    auto_renew: bool = True
    key_size: int = 2048
    validity_days: int = 365
    min_tls_version: str = "1.3"
    cipher_suites: list = field(default_factory=lambda: [
        'TLS_AES_256_GCM_SHA384',
        'TLS_CHACHA20_POLY1305_SHA256',
        'TLS_AES_128_GCM_SHA256'
    ])


@dataclass
class AuthConfig:
    """Authentication configuration settings"""
    access_token_ttl: int = 900  # 15 minutes
    refresh_token_ttl: int = 604800  # 7 days
    algorithm: str = "RS256"
    max_failed_attempts: int = 5
    lockout_duration: int = 900  # 15 minutes
    session_timeout: int = 3600  # 1 hour
    require_mfa: bool = False
    password_min_length: int = 8
    password_complexity: Dict[str, bool] = field(default_factory=lambda: {
        'require_uppercase': True,
        'require_lowercase': True,
        'require_digit': True,
        'require_special': True
    })


@dataclass
class RBACConfig:
    """RBAC configuration settings"""
    enable_role_hierarchy: bool = True
    cache_permissions: bool = True
    permission_cache_ttl: int = 300  # 5 minutes
    audit_permission_checks: bool = True
    default_role: str = "viewer"


@dataclass
class PIIConfig:
    """PII detection and masking configuration"""
    sensitivity_level: str = "MEDIUM"
    auto_mask: bool = True
    mask_char: str = "*"
    preserve_chars: int = 2
    hash_pii: bool = False
    custom_patterns: Dict[str, str] = field(default_factory=dict)
    excluded_fields: list = field(default_factory=list)


@dataclass
class AuditConfig:
    """Audit logging configuration"""
    retention_days: int = 2555  # 7 years
    encrypt_logs: bool = True
    verify_integrity: bool = True
    log_level: str = "INFO"
    batch_size: int = 100
    flush_interval: int = 60  # seconds
    enable_chain_verification: bool = True


@dataclass
class ValidationConfig:
    """Input validation configuration"""
    validation_level: str = "STANDARD"
    sanitize_input: bool = True
    max_input_length: int = 10000
    allowed_file_types: list = field(default_factory=lambda: [
        'json', 'txt', 'csv', 'yaml', 'yml'
    ])
    max_file_size: int = 10485760  # 10MB
    rate_limit_requests: int = 1000  # per hour
    rate_limit_window: int = 3600  # seconds


@dataclass
class SecurityConfig:
    """Main security configuration"""
    profile: SecurityProfile = SecurityProfile.PRODUCTION
    secret_key: str = field(default_factory=lambda: secrets.token_hex(32))
    tls: TLSConfig = field(default_factory=TLSConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    rbac: RBACConfig = field(default_factory=RBACConfig)
    pii: PIIConfig = field(default_factory=PIIConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    
    def __post_init__(self):
        """Apply profile-specific configurations"""
        self._apply_profile_settings()
    
    def _apply_profile_settings(self):
        """Apply security profile specific settings"""
        if self.profile == SecurityProfile.DEVELOPMENT:
            self._apply_development_settings()
        elif self.profile == SecurityProfile.TESTING:
            self._apply_testing_settings()
        elif self.profile == SecurityProfile.PRODUCTION:
            self._apply_production_settings()
        elif self.profile == SecurityProfile.HIGH_SECURITY:
            self._apply_high_security_settings()
    
    def _apply_development_settings(self):
        """Development environment settings"""
        self.tls.auto_renew = False
        self.tls.validity_days = 30
        self.auth.access_token_ttl = 3600  # 1 hour
        self.auth.max_failed_attempts = 10
        self.auth.require_mfa = False
        self.pii.sensitivity_level = "LOW"
        self.audit.retention_days = 30
        self.audit.encrypt_logs = False
        self.validation.validation_level = "PERMISSIVE"
    
    def _apply_testing_settings(self):
        """Testing environment settings"""
        self.tls.auto_renew = False
        self.tls.validity_days = 1
        self.auth.access_token_ttl = 300  # 5 minutes
        self.auth.refresh_token_ttl = 3600  # 1 hour
        self.auth.max_failed_attempts = 3
        self.pii.sensitivity_level = "HIGH"
        self.audit.retention_days = 7
        self.audit.encrypt_logs = True
        self.validation.validation_level = "STRICT"
    
    def _apply_production_settings(self):
        """Production environment settings"""
        self.tls.auto_renew = True
        self.tls.validity_days = 365
        self.auth.require_mfa = True
        self.auth.password_min_length = 12
        self.pii.sensitivity_level = "HIGH"
        self.pii.auto_mask = True
        self.audit.encrypt_logs = True
        self.audit.verify_integrity = True
        self.validation.validation_level = "STRICT"
    
    def _apply_high_security_settings(self):
        """High security environment settings"""
        self.tls.key_size = 4096
        self.tls.validity_days = 90
        self.auth.access_token_ttl = 300  # 5 minutes
        self.auth.refresh_token_ttl = 86400  # 1 day
        self.auth.max_failed_attempts = 3
        self.auth.lockout_duration = 3600  # 1 hour
        self.auth.require_mfa = True
        self.auth.password_min_length = 16
        self.pii.sensitivity_level = "STRICT"
        self.pii.hash_pii = True
        self.audit.retention_days = 3650  # 10 years
        self.audit.enable_chain_verification = True
        self.validation.validation_level = "PARANOID"
        self.validation.max_input_length = 1000
    
    @classmethod
    def from_environment(cls) -> 'SecurityConfig':
        """Create configuration from environment variables"""
        profile_name = os.getenv('ESCAI_SECURITY_PROFILE', 'PRODUCTION')
        profile = SecurityProfile(profile_name.lower())
        
        config = cls(profile=profile)
        
        # Override with environment variables if present
        if os.getenv('ESCAI_SECRET_KEY'):
            config.secret_key = os.getenv('ESCAI_SECRET_KEY')
        
        # TLS settings
        if os.getenv('ESCAI_TLS_CERT_DIR'):
            config.tls.cert_dir = os.getenv('ESCAI_TLS_CERT_DIR')
        if os.getenv('ESCAI_TLS_AUTO_RENEW'):
            config.tls.auto_renew = os.getenv('ESCAI_TLS_AUTO_RENEW').lower() == 'true'
        
        # Auth settings
        if os.getenv('ESCAI_ACCESS_TOKEN_TTL'):
            config.auth.access_token_ttl = int(os.getenv('ESCAI_ACCESS_TOKEN_TTL'))
        if os.getenv('ESCAI_REFRESH_TOKEN_TTL'):
            config.auth.refresh_token_ttl = int(os.getenv('ESCAI_REFRESH_TOKEN_TTL'))
        if os.getenv('ESCAI_REQUIRE_MFA'):
            config.auth.require_mfa = os.getenv('ESCAI_REQUIRE_MFA').lower() == 'true'
        
        # PII settings
        if os.getenv('ESCAI_PII_SENSITIVITY'):
            config.pii.sensitivity_level = os.getenv('ESCAI_PII_SENSITIVITY')
        if os.getenv('ESCAI_PII_AUTO_MASK'):
            config.pii.auto_mask = os.getenv('ESCAI_PII_AUTO_MASK').lower() == 'true'
        
        # Audit settings
        if os.getenv('ESCAI_AUDIT_RETENTION_DAYS'):
            config.audit.retention_days = int(os.getenv('ESCAI_AUDIT_RETENTION_DAYS'))
        if os.getenv('ESCAI_AUDIT_ENCRYPT'):
            config.audit.encrypt_logs = os.getenv('ESCAI_AUDIT_ENCRYPT').lower() == 'true'
        
        # Validation settings
        if os.getenv('ESCAI_VALIDATION_LEVEL'):
            config.validation.validation_level = os.getenv('ESCAI_VALIDATION_LEVEL')
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'profile': self.profile.value,
            'secret_key': '***REDACTED***',  # Never expose secret key
            'tls': {
                'cert_dir': self.tls.cert_dir,
                'auto_renew': self.tls.auto_renew,
                'key_size': self.tls.key_size,
                'validity_days': self.tls.validity_days,
                'min_tls_version': self.tls.min_tls_version,
                'cipher_suites': self.tls.cipher_suites
            },
            'auth': {
                'access_token_ttl': self.auth.access_token_ttl,
                'refresh_token_ttl': self.auth.refresh_token_ttl,
                'algorithm': self.auth.algorithm,
                'max_failed_attempts': self.auth.max_failed_attempts,
                'lockout_duration': self.auth.lockout_duration,
                'session_timeout': self.auth.session_timeout,
                'require_mfa': self.auth.require_mfa,
                'password_min_length': self.auth.password_min_length,
                'password_complexity': self.auth.password_complexity
            },
            'rbac': {
                'enable_role_hierarchy': self.rbac.enable_role_hierarchy,
                'cache_permissions': self.rbac.cache_permissions,
                'permission_cache_ttl': self.rbac.permission_cache_ttl,
                'audit_permission_checks': self.rbac.audit_permission_checks,
                'default_role': self.rbac.default_role
            },
            'pii': {
                'sensitivity_level': self.pii.sensitivity_level,
                'auto_mask': self.pii.auto_mask,
                'mask_char': self.pii.mask_char,
                'preserve_chars': self.pii.preserve_chars,
                'hash_pii': self.pii.hash_pii,
                'custom_patterns': self.pii.custom_patterns,
                'excluded_fields': self.pii.excluded_fields
            },
            'audit': {
                'retention_days': self.audit.retention_days,
                'encrypt_logs': self.audit.encrypt_logs,
                'verify_integrity': self.audit.verify_integrity,
                'log_level': self.audit.log_level,
                'batch_size': self.audit.batch_size,
                'flush_interval': self.audit.flush_interval,
                'enable_chain_verification': self.audit.enable_chain_verification
            },
            'validation': {
                'validation_level': self.validation.validation_level,
                'sanitize_input': self.validation.sanitize_input,
                'max_input_length': self.validation.max_input_length,
                'allowed_file_types': self.validation.allowed_file_types,
                'max_file_size': self.validation.max_file_size,
                'rate_limit_requests': self.validation.rate_limit_requests,
                'rate_limit_window': self.validation.rate_limit_window
            }
        }
    
    def validate_config(self) -> list:
        """Validate configuration and return any issues"""
        issues = []
        
        # Validate secret key
        if len(self.secret_key) < 32:
            issues.append("Secret key should be at least 32 characters")
        
        # Validate token TTLs
        if self.auth.access_token_ttl >= self.auth.refresh_token_ttl:
            issues.append("Access token TTL should be less than refresh token TTL")
        
        # Validate TLS settings
        if self.tls.key_size < 2048:
            issues.append("TLS key size should be at least 2048 bits")
        
        # Validate password settings
        if self.auth.password_min_length < 8:
            issues.append("Password minimum length should be at least 8 characters")
        
        # Validate audit retention
        if self.audit.retention_days < 1:
            issues.append("Audit retention should be at least 1 day")
        
        return issues


def get_security_config() -> SecurityConfig:
    """Get security configuration from environment or defaults"""
    return SecurityConfig.from_environment()