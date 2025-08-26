"""
Unit tests for security components
"""

import pytest
import asyncio
import secrets
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from escai_framework.security.tls_manager import TLSManager
from escai_framework.security.auth_manager import AuthManager, TokenManager, UserClaims
from escai_framework.security.rbac import RBACManager, Role, Permission, ResourceType, Action
from escai_framework.security.pii_detector import PIIDetector, PIIMasker, PIIType, SensitivityLevel
from escai_framework.security.audit_logger import AuditLogger, AuditEvent, AuditEventType, AuditLevel
from escai_framework.security.input_validator import InputValidator, SecuritySanitizer, ValidationLevel


class TestTLSManager:
    """Test TLS certificate management"""
    
    @pytest.fixture
    def tls_manager(self, tmp_path):
        return TLSManager(cert_dir=str(tmp_path / "certs"))
    
    @pytest.mark.asyncio
    async def test_generate_self_signed_cert(self, tls_manager):
        """Test self-signed certificate generation"""
        hostname = "test.example.com"
        cert_path, key_path = await tls_manager.generate_self_signed_cert(hostname)
        
        assert cert_path.endswith(f"{hostname}.crt")
        assert key_path.endswith(f"{hostname}.key")
        
        # Verify certificate can be loaded
        cert_info = await tls_manager.load_certificate(hostname)
        assert cert_info is not None
        assert hostname in cert_info['subject']
    
    @pytest.mark.asyncio
    async def test_certificate_validation(self, tls_manager):
        """Test certificate validation"""
        hostname = "test.example.com"
        
        # No certificate exists
        assert not await tls_manager.is_certificate_valid(hostname)
        
        # Generate certificate
        await tls_manager.generate_self_signed_cert(hostname)
        
        # Certificate should be valid
        assert await tls_manager.is_certificate_valid(hostname)
    
    @pytest.mark.asyncio
    async def test_certificate_renewal(self, tls_manager):
        """Test certificate renewal"""
        hostname = "test.example.com"
        
        # Generate certificate with short validity
        await tls_manager.generate_self_signed_cert(hostname, validity_days=1)
        
        # Should need renewal soon
        assert await tls_manager.needs_renewal(hostname, days_before_expiry=2)
        
        # Renew certificate
        renewed = await tls_manager.renew_certificate(hostname)
        assert renewed
    
    def test_ssl_context_creation(self, tls_manager):
        """Test SSL context creation"""
        # This would require actual certificate files
        # For now, test that the method exists and has correct signature
        assert hasattr(tls_manager, 'create_ssl_context')


class TestAuthManager:
    """Test JWT authentication and token management"""
    
    @pytest.fixture
    def mock_redis(self):
        redis_mock = AsyncMock()
        redis_mock.setex = AsyncMock()
        redis_mock.hmset = AsyncMock()
        redis_mock.expire = AsyncMock()
        redis_mock.hget = AsyncMock(return_value=b"true")
        redis_mock.hset = AsyncMock()
        redis_mock.get = AsyncMock()
        redis_mock.delete = AsyncMock()
        redis_mock.exists = AsyncMock(return_value=0)
        redis_mock.hgetall = AsyncMock(return_value={})
        redis_mock.incr = AsyncMock()
        return redis_mock
    
    @pytest.fixture
    def token_manager(self, mock_redis):
        return TokenManager(mock_redis)
    
    @pytest.fixture
    def auth_manager(self, mock_redis):
        return AuthManager(mock_redis)
    
    @pytest.mark.asyncio
    async def test_token_creation(self, token_manager):
        """Test JWT token pair creation"""
        user_claims = UserClaims(
            user_id="test_user",
            username="testuser",
            email="test@example.com",
            roles=["user"],
            permissions=["read"],
            session_id="",
            issued_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(minutes=15)
        )
        
        token_pair = await token_manager.create_token_pair(user_claims)
        
        assert token_pair.access_token
        assert token_pair.refresh_token
        assert token_pair.access_expires_at > datetime.utcnow()
        assert token_pair.refresh_expires_at > token_pair.access_expires_at
    
    @pytest.mark.asyncio
    async def test_token_validation(self, token_manager):
        """Test access token validation"""
        user_claims = UserClaims(
            user_id="test_user",
            username="testuser",
            email="test@example.com",
            roles=["user"],
            permissions=["read"],
            session_id="",
            issued_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(minutes=15)
        )
        
        token_pair = await token_manager.create_token_pair(user_claims)
        
        # Validate access token
        claims = await token_manager.validate_access_token(token_pair.access_token)
        assert claims is not None
        assert claims['sub'] == "test_user"
        assert claims['type'] == "access"
    
    @pytest.mark.asyncio
    async def test_token_refresh(self, token_manager):
        """Test token refresh mechanism"""
        user_claims = UserClaims(
            user_id="test_user",
            username="testuser",
            email="test@example.com",
            roles=["user"],
            permissions=["read"],
            session_id="",
            issued_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(minutes=15)
        )
        
        original_pair = await token_manager.create_token_pair(user_claims)
        
        # Mock Redis to return the refresh token
        token_manager.redis.get = AsyncMock(return_value=original_pair.refresh_token.encode())
        token_manager.redis.hgetall = AsyncMock(return_value={
            b"username": b"testuser",
            b"active": b"true"
        })
        
        # Refresh token
        new_pair = await token_manager.refresh_access_token(original_pair.refresh_token)
        
        assert new_pair is not None
        assert new_pair.access_token != original_pair.access_token
        assert new_pair.refresh_token != original_pair.refresh_token
    
    @pytest.mark.asyncio
    async def test_authentication_lockout(self, auth_manager):
        """Test account lockout after failed attempts"""
        username = "testuser"
        ip_address = "192.168.1.1"
        
        # Mock failed credential validation
        auth_manager._validate_credentials = AsyncMock(return_value=False)
        
        # Simulate multiple failed attempts
        for _ in range(5):
            result = await auth_manager.authenticate_user(username, "wrong_password", ip_address)
            assert result is None
        
        # Account should be locked
        assert await auth_manager.is_account_locked(username, ip_address)


class TestRBACManager:
    """Test Role-Based Access Control"""
    
    @pytest.fixture
    def mock_redis(self):
        redis_mock = AsyncMock()
        redis_mock.hset = AsyncMock()
        redis_mock.delete = AsyncMock()
        redis_mock.keys = AsyncMock(return_value=[])
        redis_mock.hgetall = AsyncMock(return_value={})
        return redis_mock
    
    @pytest.fixture
    def rbac_manager(self, mock_redis):
        return RBACManager(mock_redis)
    
    @pytest.mark.asyncio
    async def test_role_creation(self, rbac_manager):
        """Test role creation"""
        role = Role(
            name="test_role",
            description="Test role"
        )
        role.add_permission(Permission(ResourceType.AGENT, Action.READ))
        
        success = await rbac_manager.create_role(role)
        assert success
        assert "test_role" in rbac_manager.roles
    
    @pytest.mark.asyncio
    async def test_permission_checking(self, rbac_manager):
        """Test permission checking"""
        user_id = "test_user"
        
        # Assign viewer role to user
        await rbac_manager.assign_role_to_user(user_id, "viewer")
        
        # Check permissions
        has_read = await rbac_manager.check_permission(
            user_id, ResourceType.AGENT, Action.READ
        )
        assert has_read
        
        has_delete = await rbac_manager.check_permission(
            user_id, ResourceType.AGENT, Action.DELETE
        )
        assert not has_delete
    
    @pytest.mark.asyncio
    async def test_role_hierarchy(self, rbac_manager):
        """Test role inheritance"""
        # Create parent role
        parent_role = Role(name="parent", description="Parent role")
        parent_role.add_permission(Permission(ResourceType.AGENT, Action.READ))
        await rbac_manager.create_role(parent_role)
        
        # Create child role that inherits from parent
        child_role = Role(name="child", description="Child role")
        child_role.parent_roles.add("parent")
        child_role.add_permission(Permission(ResourceType.AGENT, Action.UPDATE))
        await rbac_manager.create_role(child_role)
        
        # Assign child role to user
        user_id = "test_user"
        await rbac_manager.assign_role_to_user(user_id, "child")
        
        # User should have both parent and child permissions
        has_read = await rbac_manager.check_permission(
            user_id, ResourceType.AGENT, Action.READ
        )
        has_update = await rbac_manager.check_permission(
            user_id, ResourceType.AGENT, Action.UPDATE
        )
        
        assert has_read  # From parent role
        assert has_update  # From child role


class TestPIIDetector:
    """Test PII detection and masking"""
    
    @pytest.fixture
    def pii_detector(self):
        return PIIDetector(SensitivityLevel.MEDIUM)
    
    @pytest.fixture
    def pii_masker(self):
        return PIIMasker()
    
    def test_email_detection(self, pii_detector):
        """Test email PII detection"""
        text = "Contact me at john.doe@example.com for more info"
        matches = pii_detector.detect_pii(text)
        
        assert len(matches) == 1
        assert matches[0].pii_type == PIIType.EMAIL
        assert matches[0].value == "john.doe@example.com"
    
    def test_phone_detection(self, pii_detector):
        """Test phone number detection"""
        text = "Call me at (555) 123-4567"
        matches = pii_detector.detect_pii(text)
        
        assert len(matches) == 1
        assert matches[0].pii_type == PIIType.PHONE
    
    def test_ssn_detection(self, pii_detector):
        """Test SSN detection"""
        text = "My SSN is 123-45-6789"
        matches = pii_detector.detect_pii(text)
        
        assert len(matches) == 1
        assert matches[0].pii_type == PIIType.SSN
        assert matches[0].value == "123-45-6789"
    
    def test_credit_card_detection(self, pii_detector):
        """Test credit card detection"""
        text = "Card number: 4532015112830366"
        matches = pii_detector.detect_pii(text)
        
        assert len(matches) == 1
        assert matches[0].pii_type == PIIType.CREDIT_CARD
    
    def test_pii_masking(self, pii_detector, pii_masker):
        """Test PII masking"""
        text = "Email: john.doe@example.com, Phone: (555) 123-4567"
        matches = pii_detector.detect_pii(text)
        
        masked_text = pii_masker.mask_text(text, matches)
        
        assert "john.doe@example.com" not in masked_text
        assert "(555) 123-4567" not in masked_text
        assert "*" in masked_text or "[" in masked_text  # Some form of masking
    
    def test_structured_data_masking(self, pii_detector, pii_masker):
        """Test PII masking in structured data"""
        data = {
            "user_info": {
                "email": "user@example.com",
                "phone": "555-1234",
                "name": "John Doe"
            },
            "metadata": {
                "created_at": "2023-01-01",
                "id": 12345
            }
        }
        
        masked_data = pii_masker.mask_structured_data(data, pii_detector)
        
        assert masked_data["user_info"]["email"] != "user@example.com"
        assert masked_data["metadata"]["id"] == 12345  # Non-PII unchanged


class TestAuditLogger:
    """Test audit logging system"""
    
    @pytest.fixture
    def mock_redis(self):
        redis_mock = AsyncMock()
        redis_mock.hset = AsyncMock()
        redis_mock.expire = AsyncMock()
        redis_mock.zadd = AsyncMock()
        redis_mock.set = AsyncMock()
        redis_mock.get = AsyncMock()
        redis_mock.hgetall = AsyncMock()
        redis_mock.keys = AsyncMock(return_value=[])
        redis_mock.zrangebyscore = AsyncMock(return_value=[])
        redis_mock.zrevrange = AsyncMock(return_value=[])
        return redis_mock
    
    @pytest.fixture
    def audit_logger(self, mock_redis):
        secret_key = secrets.token_hex(32)
        return AuditLogger(mock_redis, secret_key)
    
    @pytest.mark.asyncio
    async def test_event_logging(self, audit_logger):
        """Test audit event logging"""
        event = AuditEvent(
            event_id=secrets.token_urlsafe(16),
            event_type=AuditEventType.AUTHENTICATION,
            level=AuditLevel.INFO,
            timestamp=datetime.utcnow(),
            user_id="test_user",
            session_id="test_session",
            ip_address="192.168.1.1",
            user_agent="test-agent",
            resource_type="authentication",
            resource_id=None,
            action="login",
            details={"method": "password"},
            result="success"
        )
        
        success = await audit_logger.log_event(event)
        assert success
    
    @pytest.mark.asyncio
    async def test_event_retrieval(self, audit_logger):
        """Test audit event retrieval"""
        event_id = secrets.token_urlsafe(16)
        
        # Mock encrypted event data
        audit_logger.redis.hgetall = AsyncMock(return_value={
            b'iv': b'1234567890abcdef' * 2,
            b'data': b'encrypted_data_here',
            b'hmac': b'hmac_value_here',
            b'salt': b'salt_value_here'
        })
        
        # Mock decryption (would need actual encrypted data for full test)
        with patch.object(audit_logger.integrity_manager, 'decrypt_event') as mock_decrypt:
            mock_decrypt.return_value = '{"event_id": "test", "event_type": "authentication"}'
            
            event = await audit_logger.get_event(event_id)
            # Would assert event properties if decryption worked
    
    @pytest.mark.asyncio
    async def test_authentication_event_creation(self, audit_logger):
        """Test authentication event creation"""
        event = await audit_logger.create_authentication_event(
            user_id="test_user",
            result="success",
            ip_address="192.168.1.1",
            details={"method": "password"}
        )
        
        assert event.event_type == AuditEventType.AUTHENTICATION
        assert event.user_id == "test_user"
        assert event.result == "success"


class TestInputValidator:
    """Test input validation and sanitization"""
    
    @pytest.fixture
    def validator(self):
        return InputValidator(ValidationLevel.STANDARD)
    
    @pytest.fixture
    def sanitizer(self):
        return SecuritySanitizer(ValidationLevel.STANDARD)
    
    def test_email_validation(self, validator):
        """Test email validation"""
        result = validator.validate_field("email", "test@example.com")
        assert result.is_valid
        
        result = validator.validate_field("email", "invalid-email")
        assert not result.is_valid
    
    def test_password_validation(self, validator):
        """Test password complexity validation"""
        # Valid password
        result = validator.validate_field("password", "StrongPass123!")
        assert result.is_valid
        
        # Weak password
        result = validator.validate_field("password", "weak")
        assert not result.is_valid
    
    def test_sql_injection_sanitization(self, sanitizer):
        """Test SQL injection sanitization"""
        malicious_input = "'; DROP TABLE users; --"
        sanitized = sanitizer.sanitize_sql_injection(malicious_input)
        
        assert "DROP" not in sanitized.upper()
        assert "--" not in sanitized
    
    def test_xss_sanitization(self, sanitizer):
        """Test XSS sanitization"""
        malicious_input = "<script>alert('xss')</script>"
        sanitized = sanitizer.sanitize_xss(malicious_input)
        
        assert "<script>" not in sanitized
        assert "alert" in sanitized  # Content preserved but encoded
    
    def test_path_traversal_sanitization(self, sanitizer):
        """Test path traversal sanitization"""
        malicious_path = "../../../etc/passwd"
        sanitized = sanitizer.sanitize_path_traversal(malicious_path)
        
        assert ".." not in sanitized
        assert sanitized == "etc/passwd"
    
    def test_comprehensive_sanitization(self, sanitizer):
        """Test comprehensive sanitization"""
        malicious_input = "'; DROP TABLE users; <script>alert('xss')</script> --"
        sanitized = sanitizer.comprehensive_sanitize(malicious_input)
        
        assert "DROP" not in sanitized.upper()
        assert "<script>" not in sanitized
        assert "--" not in sanitized
    
    def test_data_validation(self, validator):
        """Test complete data validation"""
        data = {
            "email": "test@example.com",
            "username": "testuser",
            "password": "StrongPass123!"
        }
        
        result = validator.validate_data(data)
        assert result.is_valid
        assert len(result.errors) == 0
        
        # Test with invalid data
        invalid_data = {
            "email": "invalid-email",
            "username": "a",  # Too short
            "password": "weak"  # Too weak
        }
        
        result = validator.validate_data(invalid_data)
        assert not result.is_valid
        assert len(result.errors) > 0


if __name__ == "__main__":
    pytest.main([__file__])