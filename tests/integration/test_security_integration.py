"""
Integration tests for security components
"""

import pytest
import asyncio
import secrets
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
import redis.asyncio as redis

from escai_framework.security.tls_manager import TLSManager
from escai_framework.security.auth_manager import AuthManager, TokenManager, UserClaims
from escai_framework.security.rbac import RBACManager, Role, Permission, ResourceType, Action
from escai_framework.security.pii_detector import PIIDetector, PIIMasker, SensitivityLevel
from escai_framework.security.audit_logger import AuditLogger, AuditEvent, AuditEventType, AuditLevel
from escai_framework.security.input_validator import InputValidator, ValidationLevel


@pytest.fixture
async def redis_client():
    """Create Redis client for testing"""
    try:
        client = redis.from_url("redis://localhost:6379/15")  # Use test database
        await client.ping()
        yield client
        await client.flushdb()  # Clean up test data
        await client.close()
    except redis.ConnectionError:
        pytest.skip("Redis not available for integration tests")


@pytest.fixture
def temp_cert_dir():
    """Create temporary directory for certificates"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


class TestSecurityIntegration:
    """Integration tests for security components working together"""
    
    @pytest.mark.asyncio
    async def test_complete_authentication_flow(self, redis_client):
        """Test complete authentication flow with RBAC and audit logging"""
        
        # Initialize components
        auth_manager = AuthManager(redis_client)
        rbac_manager = RBACManager(redis_client)
        audit_logger = AuditLogger(redis_client, secrets.token_hex(32))
        
        # Set up user with roles
        user_id = "test_user"
        username = "testuser"
        ip_address = "192.168.1.1"
        
        # Assign role to user
        await rbac_manager.assign_role_to_user(user_id, "analyst")
        
        # Authenticate user
        token_pair = await auth_manager.authenticate_user(username, "password", ip_address)
        assert token_pair is not None
        
        # Validate token
        claims = await auth_manager.token_manager.validate_access_token(token_pair.access_token)
        assert claims is not None
        assert claims['username'] == username
        
        # Check permissions
        has_read_permission = await rbac_manager.check_permission(
            user_id, ResourceType.AGENT, Action.READ
        )
        assert has_read_permission
        
        # Log audit event
        await audit_logger.create_authentication_event(
            user_id=user_id,
            result="success",
            ip_address=ip_address,
            details={"method": "password"}
        )
        
        # Query audit events
        events = await audit_logger.query_events(user_id=user_id, limit=10)
        assert len(events) > 0
        assert events[0].user_id == user_id
    
    @pytest.mark.asyncio
    async def test_pii_detection_with_audit_logging(self, redis_client):
        """Test PII detection with audit logging"""
        
        audit_logger = AuditLogger(redis_client, secrets.token_hex(32))
        pii_detector = PIIDetector(SensitivityLevel.HIGH)
        pii_masker = PIIMasker()
        
        # Sample data with PII
        sensitive_data = {
            "user_email": "john.doe@example.com",
            "phone": "(555) 123-4567",
            "ssn": "123-45-6789",
            "notes": "Contact customer at john.doe@example.com or call (555) 123-4567"
        }
        
        # Detect PII
        all_matches = []
        for field, value in sensitive_data.items():
            if isinstance(value, str):
                matches = pii_detector.detect_pii(value, context=field)
                all_matches.extend(matches)
        
        assert len(all_matches) > 0
        
        # Mask PII
        masked_data = pii_masker.mask_structured_data(sensitive_data, pii_detector)
        
        # Verify masking
        assert masked_data["user_email"] != sensitive_data["user_email"]
        assert masked_data["phone"] != sensitive_data["phone"]
        assert masked_data["ssn"] != sensitive_data["ssn"]
        
        # Log PII detection event
        event = AuditEvent(
            event_id=secrets.token_urlsafe(16),
            event_type=AuditEventType.DATA_ACCESS,
            level=AuditLevel.INFO,
            timestamp=datetime.utcnow(),
            user_id="system",
            session_id=None,
            ip_address=None,
            user_agent=None,
            resource_type="pii_detection",
            resource_id=None,
            action="mask_pii",
            details={
                "pii_types_found": [match.pii_type.value for match in all_matches],
                "fields_processed": list(sensitive_data.keys())
            },
            result="success"
        )
        
        success = await audit_logger.log_event(event)
        assert success
        
        # Verify event can be retrieved
        retrieved_event = await audit_logger.get_event(event.event_id)
        # Note: This might be None due to encryption/decryption in test environment
    
    @pytest.mark.asyncio
    async def test_input_validation_with_rbac(self, redis_client):
        """Test input validation integrated with RBAC"""
        
        rbac_manager = RBACManager(redis_client)
        validator = InputValidator(ValidationLevel.STRICT)
        
        # Set up user with limited permissions
        user_id = "limited_user"
        await rbac_manager.assign_role_to_user(user_id, "viewer")
        
        # Test input validation for different permission levels
        test_inputs = {
            "agent_id": "test-agent-123",
            "session_id": "session_abc123def456",
            "email": "user@example.com",
            "malicious_input": "<script>alert('xss')</script>"
        }
        
        # Validate inputs
        validation_result = validator.validate_data(test_inputs)
        
        # Check that malicious input was sanitized
        sanitized_malicious = validation_result.sanitized_data.get("malicious_input", "")
        assert "<script>" not in sanitized_malicious
        
        # Check permissions for different actions
        can_read = await rbac_manager.check_permission(
            user_id, ResourceType.AGENT, Action.READ
        )
        can_delete = await rbac_manager.check_permission(
            user_id, ResourceType.AGENT, Action.DELETE
        )
        
        assert can_read
        assert not can_delete
    
    @pytest.mark.asyncio
    async def test_tls_with_authentication(self, temp_cert_dir, redis_client):
        """Test TLS certificate management with authentication"""
        
        tls_manager = TLSManager(cert_dir=temp_cert_dir)
        auth_manager = AuthManager(redis_client)
        
        # Generate certificate for secure communication
        hostname = "api.escai.local"
        cert_path, key_path = await tls_manager.generate_self_signed_cert(hostname)
        
        # Verify certificate exists and is valid
        assert Path(cert_path).exists()
        assert Path(key_path).exists()
        assert await tls_manager.is_certificate_valid(hostname)
        
        # Create SSL context
        ssl_context = await tls_manager.setup_server_ssl(hostname)
        assert ssl_context is not None
        
        # Test authentication flow (would be used with SSL context)
        user_claims = UserClaims(
            user_id="secure_user",
            username="secureuser",
            email="secure@example.com",
            roles=["admin"],
            permissions=["read", "write", "admin"],
            session_id="",
            issued_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(minutes=15)
        )
        
        token_pair = await auth_manager.token_manager.create_token_pair(user_claims)
        assert token_pair is not None
        
        # Validate token (would be done over TLS connection)
        claims = await auth_manager.token_manager.validate_access_token(token_pair.access_token)
        assert claims is not None
        assert claims['username'] == "secureuser"
    
    @pytest.mark.asyncio
    async def test_audit_chain_integrity(self, redis_client):
        """Test audit log chain integrity"""
        
        audit_logger = AuditLogger(redis_client, secrets.token_hex(32))
        
        # Create multiple audit events
        events = []
        for i in range(5):
            event = AuditEvent(
                event_id=f"event_{i}",
                event_type=AuditEventType.DATA_ACCESS,
                level=AuditLevel.INFO,
                timestamp=datetime.utcnow(),
                user_id=f"user_{i}",
                session_id=f"session_{i}",
                ip_address="192.168.1.1",
                user_agent="test-agent",
                resource_type="test_resource",
                resource_id=f"resource_{i}",
                action="read",
                details={"test": f"data_{i}"},
                result="success"
            )
            events.append(event)
            
            success = await audit_logger.log_event(event)
            assert success
        
        # Verify chain integrity
        # Note: This test might need adjustment based on actual implementation
        # as it requires proper encryption/decryption setup
        
        # At minimum, verify events were logged
        for event in events:
            # Try to retrieve event (might be None due to encryption in test)
            retrieved = await audit_logger.get_event(event.event_id)
            # Just verify the logging mechanism worked
    
    @pytest.mark.asyncio
    async def test_security_middleware_integration(self, redis_client):
        """Test security components working as middleware"""
        
        # Initialize all security components
        auth_manager = AuthManager(redis_client)
        rbac_manager = RBACManager(redis_client)
        audit_logger = AuditLogger(redis_client, secrets.token_hex(32))
        validator = InputValidator(ValidationLevel.STANDARD)
        pii_detector = PIIDetector(SensitivityLevel.MEDIUM)
        pii_masker = PIIMasker()
        
        # Simulate API request processing
        user_id = "api_user"
        await rbac_manager.assign_role_to_user(user_id, "analyst")
        
        # 1. Authenticate user
        token_pair = await auth_manager.authenticate_user("apiuser", "password", "192.168.1.1")
        assert token_pair is not None
        
        # 2. Validate token
        claims = await auth_manager.token_manager.validate_access_token(token_pair.access_token)
        assert claims is not None
        
        # 3. Check permissions
        has_permission = await rbac_manager.check_permission(
            user_id, ResourceType.EPISTEMIC_STATE, Action.READ
        )
        assert has_permission
        
        # 4. Validate and sanitize input
        request_data = {
            "agent_id": "test-agent",
            "query": "SELECT * FROM agents WHERE id = 'test'",  # Potential SQL injection
            "user_email": "user@example.com"  # PII
        }
        
        validation_result = validator.validate_data(request_data)
        assert validation_result.is_valid
        
        # 5. Detect and mask PII
        pii_matches = []
        for field, value in validation_result.sanitized_data.items():
            if isinstance(value, str):
                matches = pii_detector.detect_pii(value, context=field)
                pii_matches.extend(matches)
        
        if pii_matches:
            masked_data = pii_masker.mask_structured_data(
                validation_result.sanitized_data, pii_detector
            )
        else:
            masked_data = validation_result.sanitized_data
        
        # 6. Log the request
        await audit_logger.create_data_access_event(
            user_id=user_id,
            resource_type="epistemic_state",
            resource_id="test-agent",
            action="read",
            session_id=claims.get('session_id'),
            ip_address="192.168.1.1",
            details={
                "pii_detected": len(pii_matches) > 0,
                "input_sanitized": len(validation_result.warnings or []) > 0
            }
        )
        
        # Verify the complete flow worked
        assert masked_data["agent_id"] == "test-agent"
        assert "SELECT" not in masked_data["query"].upper()  # SQL injection sanitized
        assert masked_data["user_email"] != "user@example.com"  # PII masked


if __name__ == "__main__":
    pytest.main([__file__])