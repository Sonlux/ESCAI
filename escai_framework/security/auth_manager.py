"""
Enhanced JWT Authentication with Refresh Token Mechanisms

Provides secure JWT token management with refresh tokens, secure storage,
token rotation, and comprehensive security features.
"""

import jwt
import secrets
import hashlib
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any, cast, Union
from dataclasses import dataclass, asdict
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
import redis.asyncio as redis
import logging

logger = logging.getLogger(__name__)


@dataclass
class TokenPair:
    """JWT token pair with access and refresh tokens"""
    access_token: str
    refresh_token: str
    access_expires_at: datetime
    refresh_expires_at: datetime
    token_type: str = "Bearer"


@dataclass
class UserClaims:
    """User claims for JWT tokens"""
    user_id: str
    username: str
    email: str
    roles: List[str]
    permissions: List[str]
    session_id: str
    issued_at: datetime
    expires_at: datetime


class TokenManager:
    """Manages JWT token generation, validation, and storage"""
    
    def __init__(
        self,
        redis_client: redis.Redis,
        access_token_ttl: int = 900,  # 15 minutes
        refresh_token_ttl: int = 604800,  # 7 days
        algorithm: str = "RS256"
    ):
        self.redis = redis_client
        self.access_token_ttl = access_token_ttl
        self.refresh_token_ttl = refresh_token_ttl
        self.algorithm = algorithm
        self.private_key = None
        self.public_key = None
        self._generate_keys()
    
    def _generate_keys(self):
        """Generate RSA key pair for JWT signing"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        self.private_key = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        self.public_key = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
    
    async def create_token_pair(self, user_claims: UserClaims) -> TokenPair:
        """Create access and refresh token pair"""
        
        now = datetime.utcnow()
        session_id = secrets.token_urlsafe(32)
        
        # Access token claims
        access_claims = {
            "sub": user_claims.user_id,
            "username": user_claims.username,
            "email": user_claims.email,
            "roles": user_claims.roles,
            "permissions": user_claims.permissions,
            "session_id": session_id,
            "iat": now,
            "exp": now + timedelta(seconds=self.access_token_ttl),
            "type": "access"
        }
        
        # Refresh token claims
        refresh_claims = {
            "sub": user_claims.user_id,
            "session_id": session_id,
            "iat": now,
            "exp": now + timedelta(seconds=self.refresh_token_ttl),
            "type": "refresh"
        }
        
        # Generate tokens
        access_token = jwt.encode(access_claims, self.private_key, algorithm=self.algorithm)
        refresh_token = jwt.encode(refresh_claims, self.private_key, algorithm=self.algorithm)
        
        # Store refresh token in Redis with expiration
        refresh_key = f"refresh_token:{session_id}"
        setex_result = self.redis.setex(
            refresh_key,
            self.refresh_token_ttl,
            refresh_token
        )
        if hasattr(setex_result, '__await__'):
            await setex_result
        
        # Store session info
        session_key = f"session:{session_id}"
        session_data: Dict[str, Union[str, int, float, bytes]] = {
            "user_id": user_claims.user_id,
            "username": user_claims.username,
            "created_at": now.isoformat(),
            "last_activity": now.isoformat(),
            "active": "true"
        }
        hset_result = self.redis.hset(session_key, mapping=cast(Dict[str, Union[str, int, float, bytes]], session_data))
        if hasattr(hset_result, '__await__'):
            await hset_result
        expire_result = self.redis.expire(session_key, self.refresh_token_ttl)
        if hasattr(expire_result, '__await__'):
            await expire_result
        
        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            access_expires_at=cast(datetime, access_claims["exp"]),
            refresh_expires_at=cast(datetime, refresh_claims["exp"])
        )
    
    async def validate_access_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate access token and return claims"""
        try:
            claims = jwt.decode(token, self.public_key, algorithms=[self.algorithm])
            
            # Check token type
            if claims.get("type") != "access":
                return None
            
            # Check if session is still active
            session_id = claims.get("session_id")
            if session_id:
                session_key = f"session:{session_id}"
                hget_result = self.redis.hget(session_key, "active")
                session_active = await hget_result if hasattr(hget_result, '__await__') else hget_result
                if session_active != "true":
                    return None
                
                # Update last activity
                hset_result = self.redis.hset(
                    session_key,
                    "last_activity",
                    datetime.utcnow().isoformat()
                )
                if hasattr(hset_result, '__await__'):
                    await hset_result
            
            return claims
            
        except jwt.ExpiredSignatureError:
            logger.warning("Access token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid access token: {e}")
            return None
    
    async def refresh_access_token(self, refresh_token: str) -> Optional[TokenPair]:
        """Refresh access token using refresh token"""
        try:
            # Validate refresh token
            claims = jwt.decode(refresh_token, self.public_key, algorithms=[self.algorithm])
            
            if claims.get("type") != "refresh":
                return None
            
            session_id = claims.get("session_id")
            user_id = claims.get("sub")
            
            # Check if refresh token exists in Redis
            refresh_key = f"refresh_token:{session_id}"
            get_result = self.redis.get(refresh_key)
            stored_token = await get_result if hasattr(get_result, '__await__') else get_result
            
            if not stored_token or (stored_token.decode() if hasattr(stored_token, 'decode') else str(stored_token)) != refresh_token:
                return None
            
            # Get session info
            session_key = f"session:{session_id}"
            hgetall_result = self.redis.hgetall(session_key)
            session_data = await hgetall_result if hasattr(hgetall_result, '__await__') else hgetall_result
            
            if not session_data or session_data.get("active") != "true":
                return None
            
            # Create new token pair (token rotation)
            user_claims = UserClaims(
                user_id=user_id,
                username=session_data["username"],
                email="",  # Would be fetched from user service
                roles=[],  # Would be fetched from user service
                permissions=[],  # Would be fetched from user service
                session_id=session_id,
                issued_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(seconds=self.access_token_ttl)
            )
            
            # Invalidate old refresh token
            await self.redis.delete(refresh_key)
            
            return await self.create_token_pair(user_claims)
            
        except jwt.ExpiredSignatureError:
            logger.warning("Refresh token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid refresh token: {e}")
            return None
    
    async def revoke_token(self, token: str) -> bool:
        """Revoke access or refresh token"""
        try:
            claims = jwt.decode(
                token,
                self.public_key,
                algorithms=[self.algorithm],
                options={"verify_exp": False}  # Allow expired tokens for revocation
            )
            
            session_id = claims.get("session_id")
            if not session_id:
                return False
            
            # Mark session as inactive
            session_key = f"session:{session_id}"
            hset_result = self.redis.hset(session_key, "active", "false")
            if hasattr(hset_result, '__await__'):
                await hset_result
            
            # Remove refresh token
            refresh_key = f"refresh_token:{session_id}"
            await self.redis.delete(refresh_key)
            
            # Add token to blacklist
            jti = claims.get("jti", hashlib.sha256(token.encode()).hexdigest())
            blacklist_key = f"blacklist:{jti}"
            exp = claims.get("exp", datetime.utcnow().timestamp() + 3600)
            exp_timestamp = exp if isinstance(exp, (int, float)) else exp.timestamp() if hasattr(exp, 'timestamp') else float(exp)
            ttl = max(int(exp_timestamp - datetime.utcnow().timestamp()), 0)
            
            if ttl > 0:
                setex_result = self.redis.setex(blacklist_key, ttl, "revoked")
                if hasattr(setex_result, '__await__'):
                    await setex_result
            
            return True
            
        except jwt.InvalidTokenError:
            return False
    
    async def is_token_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted"""
        try:
            claims = jwt.decode(
                token,
                self.public_key,
                algorithms=[self.algorithm],
                options={"verify_exp": False}
            )
            
            jti = claims.get("jti", hashlib.sha256(token.encode()).hexdigest())
            blacklist_key = f"blacklist:{jti}"
            
            exists_result = self.redis.exists(blacklist_key)
            exists_count = await exists_result if hasattr(exists_result, '__await__') else exists_result
            return cast(int, exists_count) > 0
            
        except jwt.InvalidTokenError:
            return True  # Invalid tokens are considered blacklisted
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions and tokens"""
        try:
            # Get all session keys
            session_keys = await self.redis.keys("session:*")
            
            for key in session_keys:
                ttl_result = self.redis.ttl(key)
                ttl = await ttl_result if hasattr(ttl_result, '__await__') else ttl_result
                if ttl <= 0:  # Expired
                    await self.redis.delete(key)
            
            # Clean up expired blacklist entries
            blacklist_keys = await self.redis.keys("blacklist:*")
            for key in blacklist_keys:
                ttl_result = self.redis.ttl(key)
                ttl = await ttl_result if hasattr(ttl_result, '__await__') else ttl_result
                if ttl <= 0:
                    await self.redis.delete(key)
                    
        except Exception as e:
            logger.error(f"Session cleanup error: {e}")


class AuthManager:
    """Main authentication manager"""
    
    def __init__(self, redis_client: redis.Redis):
        self.token_manager = TokenManager(redis_client)
        self.redis = redis_client
        self.failed_attempts: Dict[str, int] = {}
        self.lockout_duration = 900  # 15 minutes
        self.max_attempts = 5
    
    async def authenticate_user(
        self,
        username: str,
        password: str,
        ip_address: str
    ) -> Optional[TokenPair]:
        """Authenticate user and return token pair"""
        
        # Check for account lockout
        if await self.is_account_locked(username, ip_address):
            logger.warning(f"Authentication blocked for {username} from {ip_address} - account locked")
            return None
        
        # Validate credentials (would integrate with user service)
        if not await self._validate_credentials(username, password):
            await self._record_failed_attempt(username, ip_address)
            return None
        
        # Reset failed attempts on successful login
        await self._reset_failed_attempts(username, ip_address)
        
        # Get user info (would fetch from user service)
        user_claims = UserClaims(
            user_id=f"user_{username}",
            username=username,
            email=f"{username}@example.com",
            roles=["user"],
            permissions=["read", "write"],
            session_id="",  # Will be set by token manager
            issued_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(seconds=900)
        )
        
        return await self.token_manager.create_token_pair(user_claims)
    
    async def _validate_credentials(self, username: str, password: str) -> bool:
        """Validate user credentials (placeholder - integrate with user service)"""
        # This would integrate with your user authentication service
        # For demo purposes, accept any non-empty credentials
        return bool(username and password)
    
    async def is_account_locked(self, username: str, ip_address: str) -> bool:
        """Check if account is locked due to failed attempts"""
        user_key = f"failed_attempts:user:{username}"
        ip_key = f"failed_attempts:ip:{ip_address}"
        
        user_attempts = await self.redis.get(user_key)
        ip_attempts = await self.redis.get(ip_key)
        
        user_count = int(user_attempts) if user_attempts else 0
        ip_count = int(ip_attempts) if ip_attempts else 0
        
        return user_count >= self.max_attempts or ip_count >= self.max_attempts
    
    async def _record_failed_attempt(self, username: str, ip_address: str):
        """Record failed authentication attempt"""
        user_key = f"failed_attempts:user:{username}"
        ip_key = f"failed_attempts:ip:{ip_address}"
        
        # Increment counters with expiration
        await self.redis.incr(user_key)
        await self.redis.expire(user_key, self.lockout_duration)
        
        await self.redis.incr(ip_key)
        await self.redis.expire(ip_key, self.lockout_duration)
    
    async def _reset_failed_attempts(self, username: str, ip_address: str):
        """Reset failed attempt counters"""
        user_key = f"failed_attempts:user:{username}"
        ip_key = f"failed_attempts:ip:{ip_address}"
        
        await self.redis.delete(user_key)
        await self.redis.delete(ip_key)
    
    async def start_cleanup_task(self):
        """Start background cleanup task"""
        async def cleanup_task():
            while True:
                try:
                    await self.token_manager.cleanup_expired_sessions()
                    await asyncio.sleep(3600)  # Run every hour
                except Exception as e:
                    logger.error(f"Cleanup task error: {e}")
                    await asyncio.sleep(300)  # Retry in 5 minutes
        
        asyncio.create_task(cleanup_task())
        logger.info("Started authentication cleanup task")