# ESCAI Framework Security Guide

## Overview

The ESCAI Framework implements comprehensive security measures to protect sensitive agent execution data and ensure secure operation in production environments. This guide covers all security features and best practices for deployment and configuration.

## Security Components

### 1. TLS 1.3 Encryption

**Purpose**: Secure all data transmission with the latest TLS encryption standards.

**Features**:

- Automatic certificate generation and management
- Certificate renewal and rotation
- TLS 1.3 enforcement with secure cipher suites
- Hostname verification and certificate validation

**Configuration**:

```python
from escai_framework.security import TLSManager

tls_manager = TLSManager(
    cert_dir="certs",
    auto_renew=True
)

# Generate certificate
cert_path, key_path = await tls_manager.generate_self_signed_cert("api.escai.local")

# Create SSL context
ssl_context = await tls_manager.setup_server_ssl("api.escai.local")
```

### 2. JWT Authentication with Refresh Tokens

**Purpose**: Secure user authentication with token-based access control.

**Features**:

- RSA-256 signed JWT tokens
- Refresh token rotation
- Session management with Redis
- Account lockout protection
- Token blacklisting and revocation

**Configuration**:

```python
from escai_framework.security import AuthManager

auth_manager = AuthManager(redis_client)

# Authenticate user
token_pair = await auth_manager.authenticate_user(
    username="user",
    password="password",
    ip_address="192.168.1.1"
)

# Validate token
claims = await auth_manager.token_manager.validate_access_token(
    token_pair.access_token
)
```

### 3. Role-Based Access Control (RBAC)

**Purpose**: Fine-grained permission management with hierarchical roles.

**Features**:

- Predefined system roles (super_admin, admin, analyst, monitor, viewer)
- Custom role creation with specific permissions
- Role hierarchy and inheritance
- Resource-specific permissions
- Conditional permissions (time-based, IP-based, owner-only)

**Configuration**:

```python
from escai_framework.security import RBACManager, Role, Permission, ResourceType, Action

rbac_manager = RBACManager(redis_client)

# Create custom role
custom_role = Role(name="data_scientist", description="Data analysis role")
custom_role.add_permission(Permission(ResourceType.EPISTEMIC_STATE, Action.READ))
custom_role.add_permission(Permission(ResourceType.BEHAVIORAL_PATTERN, Action.READ))

await rbac_manager.create_role(custom_role)

# Assign role to user
await rbac_manager.assign_role_to_user("user_id", "data_scientist")

# Check permissions
has_permission = await rbac_manager.check_permission(
    user_id="user_id",
    resource_type=ResourceType.EPISTEMIC_STATE,
    action=Action.READ
)
```

### 4. PII Detection and Masking

**Purpose**: Automatically detect and mask personally identifiable information.

**Features**:

- Pattern-based PII detection (email, phone, SSN, credit cards, etc.)
- Configurable sensitivity levels
- Multiple masking strategies (character masking, hashing, replacement)
- Structured data processing
- Custom pattern support

**Configuration**:

```python
from escai_framework.security import PIIDetector, PIIMasker, SensitivityLevel

detector = PIIDetector(SensitivityLevel.HIGH)
masker = PIIMasker()

# Detect PII
text = "Contact John Doe at john.doe@example.com or (555) 123-4567"
matches = detector.detect_pii(text)

# Mask PII
masked_text = masker.mask_text(text, matches)
# Result: "Contact John Doe at jo**@example.com or (5**) ***-**67"

# Process structured data
data = {"email": "user@example.com", "phone": "555-1234"}
masked_data = masker.mask_structured_data(data, detector)
```

### 5. Comprehensive Audit Logging

**Purpose**: Tamper-proof audit logging for compliance and security monitoring.

**Features**:

- Cryptographic integrity protection
- Blockchain-style hash chaining
- Encrypted log storage
- Multiple indexing strategies
- Configurable retention policies
- Audit trail verification

**Configuration**:

```python
from escai_framework.security import AuditLogger, AuditEventType

audit_logger = AuditLogger(redis_client, secret_key)

# Log authentication event
await audit_logger.create_authentication_event(
    user_id="user_id",
    result="success",
    ip_address="192.168.1.1",
    details={"method": "password"}
)

# Query audit events
events = await audit_logger.query_events(
    user_id="user_id",
    start_time=datetime.now() - timedelta(days=7)
)

# Verify chain integrity
integrity_ok = await audit_logger.verify_chain_integrity()
```

### 6. Input Validation and Sanitization

**Purpose**: Protect against injection attacks and malicious input.

**Features**:

- SQL injection prevention
- XSS protection
- Path traversal prevention
- Command injection protection
- LDAP injection protection
- Configurable validation levels
- Custom validation rules

**Configuration**:

```python
from escai_framework.security import InputValidator, ValidationLevel

validator = InputValidator(ValidationLevel.STRICT)

# Validate individual field
result = validator.validate_field("email", "user@example.com")

# Validate complete data
data = {
    "username": "testuser",
    "email": "user@example.com",
    "query": "SELECT * FROM users"  # Will be sanitized
}

validation_result = validator.validate_data(data)
if validation_result.is_valid:
    safe_data = validation_result.sanitized_data
```

## Security Configuration

### Environment Variables

Configure security settings using environment variables:

```bash
# Security profile
export ESCAI_SECURITY_PROFILE=production

# Secret key (generate with: python -c "import secrets; print(secrets.token_hex(32))")
export ESCAI_SECRET_KEY=your_secret_key_here

# TLS settings
export ESCAI_TLS_CERT_DIR=/etc/escai/certs
export ESCAI_TLS_AUTO_RENEW=true

# Authentication settings
export ESCAI_ACCESS_TOKEN_TTL=900
export ESCAI_REFRESH_TOKEN_TTL=604800
export ESCAI_REQUIRE_MFA=true

# PII settings
export ESCAI_PII_SENSITIVITY=HIGH
export ESCAI_PII_AUTO_MASK=true

# Audit settings
export ESCAI_AUDIT_RETENTION_DAYS=2555
export ESCAI_AUDIT_ENCRYPT=true

# Validation settings
export ESCAI_VALIDATION_LEVEL=STRICT
```

### Security Profiles

Choose from predefined security profiles:

- **Development**: Relaxed settings for development
- **Testing**: Strict settings for testing
- **Production**: Balanced settings for production
- **High Security**: Maximum security for sensitive environments

```python
from escai_framework.security.config import SecurityConfig, SecurityProfile

# Use specific profile
config = SecurityConfig(profile=SecurityProfile.HIGH_SECURITY)

# Load from environment
config = SecurityConfig.from_environment()
```

## Best Practices

### 1. Certificate Management

- Use proper CA-signed certificates in production
- Implement certificate rotation policies
- Monitor certificate expiration
- Use strong key sizes (minimum 2048 bits)

### 2. Authentication Security

- Enforce strong password policies
- Implement multi-factor authentication
- Use secure session management
- Monitor failed authentication attempts
- Implement account lockout policies

### 3. Access Control

- Follow principle of least privilege
- Regularly review and audit permissions
- Use role-based access control
- Implement resource-specific permissions
- Monitor permission changes

### 4. Data Protection

- Enable PII detection and masking
- Use appropriate sensitivity levels
- Regularly update PII patterns
- Monitor data access patterns
- Implement data classification

### 5. Audit and Monitoring

- Enable comprehensive audit logging
- Monitor security events
- Implement alerting for suspicious activities
- Regularly verify audit log integrity
- Maintain proper log retention

### 6. Input Security

- Validate all input data
- Use appropriate sanitization levels
- Implement rate limiting
- Monitor for attack patterns
- Keep validation rules updated

## Security Monitoring

### Key Metrics to Monitor

1. **Authentication Failures**: Track failed login attempts
2. **Permission Denials**: Monitor unauthorized access attempts
3. **PII Detections**: Track PII exposure incidents
4. **Input Violations**: Monitor malicious input attempts
5. **Certificate Status**: Track certificate health and expiration
6. **Audit Integrity**: Verify audit log chain integrity

### Alerting Rules

Set up alerts for:

- Multiple failed authentication attempts
- Privilege escalation attempts
- PII exposure incidents
- Suspicious input patterns
- Certificate expiration warnings
- Audit log integrity failures

## Compliance Considerations

### GDPR Compliance

- PII detection and masking
- Data retention policies
- Audit logging for data access
- Right to be forgotten support

### SOX Compliance

- Comprehensive audit trails
- Access control documentation
- Change management logging
- Data integrity verification

### HIPAA Compliance

- Encryption in transit and at rest
- Access control and authentication
- Audit logging and monitoring
- Data breach detection

## Troubleshooting

### Common Issues

1. **Certificate Errors**

   - Check certificate validity and expiration
   - Verify hostname matches certificate
   - Ensure proper certificate chain

2. **Authentication Failures**

   - Check token expiration
   - Verify secret key configuration
   - Check Redis connectivity

3. **Permission Denied**

   - Verify user role assignments
   - Check permission definitions
   - Review RBAC configuration

4. **PII Detection Issues**

   - Adjust sensitivity levels
   - Update custom patterns
   - Check field exclusions

5. **Audit Log Problems**
   - Verify encryption keys
   - Check Redis storage
   - Validate chain integrity

### Debug Mode

Enable debug logging for security components:

```python
import logging

logging.getLogger('escai_framework.security').setLevel(logging.DEBUG)
```

## Security Updates

Keep security components updated:

1. Regularly update dependencies
2. Monitor security advisories
3. Apply security patches promptly
4. Review and update security configurations
5. Conduct regular security assessments

For security issues or questions, contact the ESCAI security team or file an issue in the project repository.
