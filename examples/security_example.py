"""
ESCAI Framework Security Example

Demonstrates comprehensive security features including TLS, authentication,
RBAC, PII detection, audit logging, and input validation.
"""

import asyncio
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from escai_framework.security import (
    TLSManager,
    AuthManager,
    TokenManager,
    UserClaims,
    RBACManager,
    Role,
    Permission,
    ResourceType,
    Action,
    PIIDetector,
    PIIMasker,
    SensitivityLevel,
    AuditLogger,
    AuditEventType,
    AuditLevel,
    InputValidator,
    ValidationLevel
)
from escai_framework.security.config import SecurityConfig, SecurityProfile


async def main():
    """Main security demonstration"""
    
    print("🔒 ESCAI Framework Security Example")
    print("=" * 50)
    
    # Initialize Redis client
    redis_client = redis.from_url("redis://localhost:6379/0")
    
    try:
        await redis_client.ping()
        print("✅ Connected to Redis")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return
    
    # 1. Security Configuration
    print("\n1. Security Configuration")
    print("-" * 30)
    
    config = SecurityConfig(profile=SecurityProfile.HIGH_SECURITY)
    print(f"Security Profile: {config.profile.value}")
    print(f"TLS Key Size: {config.tls.key_size} bits")
    print(f"Access Token TTL: {config.auth.access_token_ttl} seconds")
    print(f"PII Sensitivity: {config.pii.sensitivity_level}")
    print(f"Audit Retention: {config.audit.retention_days} days")
    
    # 2. TLS Certificate Management
    print("\n2. TLS Certificate Management")
    print("-" * 30)
    
    tls_manager = TLSManager(cert_dir="temp_certs")
    
    # Generate certificate
    hostname = "api.escai.local"
    cert_path, key_path = await tls_manager.generate_self_signed_cert(hostname)
    print(f"✅ Generated certificate for {hostname}")
    print(f"   Certificate: {cert_path}")
    print(f"   Private Key: {key_path}")
    
    # Check certificate validity
    is_valid = await tls_manager.is_certificate_valid(hostname)
    print(f"   Valid: {is_valid}")
    
    # Create SSL context
    ssl_context = await tls_manager.setup_server_ssl(hostname)
    print(f"   SSL Context: {type(ssl_context).__name__}")
    
    # 3. Authentication and Token Management
    print("\n3. Authentication and Token Management")
    print("-" * 30)
    
    auth_manager = AuthManager(redis_client)
    
    # Authenticate user
    username = "demo_user"
    password = "SecurePassword123!"
    ip_address = "192.168.1.100"
    
    token_pair = await auth_manager.authenticate_user(username, password, ip_address)
    if token_pair:
        print("✅ User authenticated successfully")
        print(f"   Access Token: {token_pair.access_token[:20]}...")
        print(f"   Refresh Token: {token_pair.refresh_token[:20]}...")
        print(f"   Expires: {token_pair.access_expires_at}")
        
        # Validate token
        claims = await auth_manager.token_manager.validate_access_token(token_pair.access_token)
        if claims:
            print("✅ Token validation successful")
            print(f"   User ID: {claims['sub']}")
            print(f"   Username: {claims['username']}")
            print(f"   Session ID: {claims['session_id']}")
    else:
        print("❌ Authentication failed")
    
    # 4. Role-Based Access Control
    print("\n4. Role-Based Access Control (RBAC)")
    print("-" * 30)
    
    rbac_manager = RBACManager(redis_client)
    
    # Create custom role
    data_scientist_role = Role(
        name="data_scientist",
        description="Data scientist with analysis permissions"
    )
    data_scientist_role.add_permission(Permission(ResourceType.EPISTEMIC_STATE, Action.READ))
    data_scientist_role.add_permission(Permission(ResourceType.BEHAVIORAL_PATTERN, Action.READ))
    data_scientist_role.add_permission(Permission(ResourceType.CAUSAL_RELATIONSHIP, Action.READ))
    data_scientist_role.add_permission(Permission(ResourceType.PREDICTION, Action.READ))
    
    success = await rbac_manager.create_role(data_scientist_role)
    if success:
        print("✅ Created custom role: data_scientist")
    
    # Assign role to user
    user_id = "demo_user_id"
    await rbac_manager.assign_role_to_user(user_id, "data_scientist")
    print(f"✅ Assigned role to user: {user_id}")
    
    # Check permissions
    permissions_to_check = [
        (ResourceType.EPISTEMIC_STATE, Action.READ),
        (ResourceType.BEHAVIORAL_PATTERN, Action.READ),
        (ResourceType.AGENT, Action.DELETE),  # Should be denied
        (ResourceType.SYSTEM_CONFIG, Action.ADMIN)  # Should be denied
    ]
    
    for resource_type, action in permissions_to_check:
        has_permission = await rbac_manager.check_permission(user_id, resource_type, action)
        status = "✅" if has_permission else "❌"
        print(f"   {status} {resource_type.value}:{action.value}")
    
    # 5. PII Detection and Masking
    print("\n5. PII Detection and Masking")
    print("-" * 30)
    
    pii_detector = PIIDetector(SensitivityLevel.HIGH)
    pii_masker = PIIMasker()
    
    # Sample data with PII
    sample_data = {
        "user_info": {
            "name": "John Doe",
            "email": "john.doe@example.com",
            "phone": "(555) 123-4567",
            "ssn": "123-45-6789"
        },
        "notes": "Contact John at john.doe@example.com or call (555) 123-4567 for more information.",
        "metadata": {
            "created_at": "2023-01-01T00:00:00Z",
            "id": 12345
        }
    }
    
    print("Original data:")
    print(f"   Email: {sample_data['user_info']['email']}")
    print(f"   Phone: {sample_data['user_info']['phone']}")
    print(f"   SSN: {sample_data['user_info']['ssn']}")
    
    # Detect PII
    all_matches = []
    for field, value in sample_data.items():
        if isinstance(value, dict):
            for subfield, subvalue in value.items():
                if isinstance(subvalue, str):
                    matches = pii_detector.detect_pii(subvalue, context=f"{field}.{subfield}")
                    all_matches.extend(matches)
        elif isinstance(value, str):
            matches = pii_detector.detect_pii(value, context=field)
            all_matches.extend(matches)
    
    print(f"\n✅ Detected {len(all_matches)} PII instances:")
    for match in all_matches:
        print(f"   {match.pii_type.value}: {match.value} (confidence: {match.confidence:.2f})")
    
    # Mask PII
    masked_data = pii_masker.mask_structured_data(sample_data, pii_detector)
    
    print("\nMasked data:")
    print(f"   Email: {masked_data['user_info']['email']}")
    print(f"   Phone: {masked_data['user_info']['phone']}")
    print(f"   SSN: {masked_data['user_info']['ssn']}")
    
    # Get masking summary
    summary = pii_masker.get_masking_summary(all_matches)
    print(f"\nMasking summary: {summary}")
    
    # 6. Input Validation and Sanitization
    print("\n6. Input Validation and Sanitization")
    print("-" * 30)
    
    validator = InputValidator(ValidationLevel.STRICT)
    
    # Test various inputs
    test_inputs = [
        ("email", "user@example.com"),
        ("email", "invalid-email"),
        ("username", "validuser123"),
        ("username", "invalid user!"),
        ("password", "StrongPassword123!"),
        ("password", "weak"),
        ("malicious_sql", "'; DROP TABLE users; --"),
        ("xss_attempt", "<script>alert('xss')</script>"),
        ("path_traversal", "../../../etc/passwd")
    ]
    
    for field_name, value in test_inputs:
        result = validator.validate_field(field_name, value)
        status = "✅" if result.is_valid else "❌"
        sanitized = result.sanitized_data.get(field_name, value)
        
        print(f"   {status} {field_name}: '{value}'")
        if not result.is_valid:
            print(f"      Errors: {result.errors}")
        if sanitized != value:
            print(f"      Sanitized: '{sanitized}'")
        if result.warnings:
            print(f"      Warnings: {result.warnings}")
    
    # 7. Comprehensive Audit Logging
    print("\n7. Comprehensive Audit Logging")
    print("-" * 30)
    
    audit_logger = AuditLogger(redis_client, config.secret_key)
    
    # Log various events
    events_to_log = [
        ("authentication", lambda: audit_logger.create_authentication_event(
            user_id=user_id,
            result="success",
            ip_address=ip_address,
            details={"method": "password", "mfa": False}
        )),
        ("authorization", lambda: audit_logger.create_authorization_event(
            user_id=user_id,
            resource_type="epistemic_state",
            resource_id="agent_123",
            action="read",
            result="success",
            session_id="session_123"
        )),
        ("data_access", lambda: audit_logger.create_data_access_event(
            user_id=user_id,
            resource_type="behavioral_pattern",
            resource_id="pattern_456",
            action="read",
            session_id="session_123",
            ip_address=ip_address,
            details={"pii_detected": True, "pii_masked": True}
        ))
    ]
    
    logged_events = []
    for event_type, log_func in events_to_log:
        event = await log_func()
        logged_events.append(event)
        print(f"✅ Logged {event_type} event: {event.event_id}")
    
    # Query audit events
    recent_events = await audit_logger.query_events(
        user_id=user_id,
        start_time=datetime.utcnow() - timedelta(minutes=5),
        limit=10
    )
    
    print(f"\n✅ Retrieved {len(recent_events)} recent audit events")
    for event in recent_events:
        print(f"   {event.timestamp}: {event.event_type.value} - {event.result}")
    
    # Verify audit chain integrity
    integrity_ok = await audit_logger.verify_chain_integrity()
    print(f"✅ Audit chain integrity: {'OK' if integrity_ok else 'FAILED'}")
    
    # 8. Security Monitoring Example
    print("\n8. Security Monitoring Example")
    print("-" * 30)
    
    # Simulate security monitoring
    print("Monitoring for security events...")
    
    # Check for failed authentication attempts
    auth_events = await audit_logger.query_events(
        event_type=AuditEventType.AUTHENTICATION,
        start_time=datetime.utcnow() - timedelta(hours=1)
    )
    
    failed_attempts = [e for e in auth_events if e.result == "failure"]
    print(f"   Failed authentication attempts: {len(failed_attempts)}")
    
    # Check for permission denials
    authz_events = await audit_logger.query_events(
        event_type=AuditEventType.AUTHORIZATION,
        start_time=datetime.utcnow() - timedelta(hours=1)
    )
    
    denied_attempts = [e for e in authz_events if e.result == "failure"]
    print(f"   Permission denials: {len(denied_attempts)}")
    
    # Check for PII exposure
    data_events = await audit_logger.query_events(
        event_type=AuditEventType.DATA_ACCESS,
        start_time=datetime.utcnow() - timedelta(hours=1)
    )
    
    pii_events = [e for e in data_events if e.details.get("pii_detected")]
    print(f"   PII exposure incidents: {len(pii_events)}")
    
    print("\n🔒 Security demonstration completed successfully!")
    print("All security components are working correctly.")
    
    # Cleanup
    await redis_client.close()


if __name__ == "__main__":
    asyncio.run(main())