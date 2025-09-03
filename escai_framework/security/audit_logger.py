"""
Comprehensive Audit Logging System

Provides tamper-proof audit logging for all operations with
cryptographic integrity verification and secure storage.
"""

import asyncio
import hashlib
import hmac
import json
import secrets
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, cast, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SYSTEM_CONFIGURATION = "system_configuration"
    USER_MANAGEMENT = "user_management"
    SECURITY_EVENT = "security_event"
    API_REQUEST = "api_request"
    ERROR_EVENT = "error_event"
    MONITORING_EVENT = "monitoring_event"


class AuditLevel(Enum):
    """Audit logging levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Represents an audit event"""
    event_id: str
    event_type: AuditEventType
    level: AuditLevel
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    resource_type: Optional[str]
    resource_id: Optional[str]
    action: str
    details: Dict[str, Any]
    result: str  # success, failure, error
    error_message: Optional[str] = None
    duration_ms: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        data['level'] = self.level.value
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """Create from dictionary"""
        data['event_type'] = AuditEventType(data['event_type'])
        data['level'] = AuditLevel(data['level'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class AuditIntegrityManager:
    """Manages cryptographic integrity of audit logs"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key.encode()
        self.salt = secrets.token_bytes(16)
        self._derive_keys()
    
    def _derive_keys(self):
        """Derive encryption and HMAC keys"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=64,  # 32 bytes for encryption + 32 bytes for HMAC
            salt=self.salt,
            iterations=100000,
            backend=default_backend()
        )
        
        derived_key = kdf.derive(self.secret_key)
        self.encryption_key = derived_key[:32]
        self.hmac_key = derived_key[32:]
    
    def encrypt_event(self, event_data: str) -> Dict[str, str]:
        """Encrypt audit event data"""
        # Generate random IV
        iv = secrets.token_bytes(16)
        
        # Encrypt data
        cipher = Cipher(
            algorithms.AES(self.encryption_key),
            modes.CBC(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        
        # Pad data to block size
        padded_data = self._pad_data(event_data.encode())
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        # Create HMAC for integrity
        hmac_digest = hmac.new(
            self.hmac_key,
            iv + encrypted_data,
            hashlib.sha256
        ).hexdigest()
        
        return {
            'iv': iv.hex(),
            'data': encrypted_data.hex(),
            'hmac': hmac_digest,
            'salt': self.salt.hex()
        }
    
    def decrypt_event(self, encrypted_event: Dict[str, str]) -> Optional[str]:
        """Decrypt and verify audit event data"""
        try:
            iv = bytes.fromhex(encrypted_event['iv'])
            encrypted_data = bytes.fromhex(encrypted_event['data'])
            stored_hmac = encrypted_event['hmac']
            
            # Verify HMAC
            expected_hmac = hmac.new(
                self.hmac_key,
                iv + encrypted_data,
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(stored_hmac, expected_hmac):
                logger.error("Audit log integrity verification failed")
                return None
            
            # Decrypt data
            cipher = Cipher(
                algorithms.AES(self.encryption_key),
                modes.CBC(iv),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            
            padded_data = decryptor.update(encrypted_data) + decryptor.finalize()
            data = self._unpad_data(padded_data)
            
            return data.decode()
            
        except Exception as e:
            logger.error(f"Failed to decrypt audit event: {e}")
            return None
    
    def _pad_data(self, data: bytes) -> bytes:
        """Apply PKCS7 padding"""
        block_size = 16
        padding_length = block_size - (len(data) % block_size)
        padding = bytes([padding_length] * padding_length)
        return data + padding
    
    def _unpad_data(self, padded_data: bytes) -> bytes:
        """Remove PKCS7 padding"""
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]
    
    def create_chain_hash(self, previous_hash: str, event_data: str) -> str:
        """Create blockchain-style hash chain for tamper detection"""
        combined_data = f"{previous_hash}{event_data}".encode()
        return hashlib.sha256(combined_data).hexdigest()


class AuditLogger:
    """Main audit logging system"""
    
    def __init__(
        self,
        redis_client: redis.Redis,
        secret_key: str,
        retention_days: int = 2555  # 7 years default
    ):
        self.redis = redis_client
        self.integrity_manager = AuditIntegrityManager(secret_key)
        self.retention_days = retention_days
        self.last_hash = "0" * 64  # Genesis hash
    
    async def load_last_hash(self):
        """Load the last hash from the chain"""
        try:
            last_hash = await self.redis.get("audit:last_hash")
            if last_hash:
                self.last_hash = last_hash.decode()
        except Exception as e:
            logger.error(f"Failed to load last hash: {e}")
    
    async def log_event(self, event: AuditEvent) -> bool:
        """Log audit event with integrity protection"""
        try:
            # Serialize event
            event_data = json.dumps(event.to_dict(), sort_keys=True)
            
            # Create chain hash
            chain_hash = self.integrity_manager.create_chain_hash(self.last_hash, event_data)
            
            # Encrypt event
            encrypted_event = self.integrity_manager.encrypt_event(event_data)
            encrypted_event['chain_hash'] = chain_hash
            encrypted_event['previous_hash'] = self.last_hash
            
            # Store in Redis with multiple keys for different access patterns
            event_key = f"audit:event:{event.event_id}"
            # Cast to dict for Redis hset compatibility
            from typing import Any
            redis_mapping = cast(Dict[Any, Any], encrypted_event)
            hset_result = self.redis.hset(event_key, mapping=redis_mapping)
            if hasattr(hset_result, '__await__'):
                await hset_result
            expire_result = self.redis.expire(event_key, self.retention_days * 86400)
            if hasattr(expire_result, '__await__'):
                await expire_result
            
            # Add to time-based index
            time_key = f"audit:time:{event.timestamp.strftime('%Y-%m-%d')}"
            zadd_result = self.redis.zadd(time_key, {event.event_id: event.timestamp.timestamp()})
            if hasattr(zadd_result, '__await__'):
                await zadd_result
            expire_result = self.redis.expire(time_key, self.retention_days * 86400)
            if hasattr(expire_result, '__await__'):
                await expire_result
            
            # Add to user-based index if user_id exists
            if event.user_id:
                user_key = f"audit:user:{event.user_id}"
                zadd_result = self.redis.zadd(user_key, {event.event_id: event.timestamp.timestamp()})
                if hasattr(zadd_result, '__await__'):
                    await zadd_result
                expire_result = self.redis.expire(user_key, self.retention_days * 86400)
                if hasattr(expire_result, '__await__'):
                    await expire_result
            
            # Add to type-based index
            type_key = f"audit:type:{event.event_type.value}"
            zadd_result = self.redis.zadd(type_key, {event.event_id: event.timestamp.timestamp()})
            if hasattr(zadd_result, '__await__'):
                await zadd_result
            expire_result = self.redis.expire(type_key, self.retention_days * 86400)
            if hasattr(expire_result, '__await__'):
                await expire_result
            
            # Update last hash
            self.last_hash = chain_hash
            await self.redis.set("audit:last_hash", chain_hash)
            
            logger.debug(f"Logged audit event: {event.event_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            return False
    
    async def get_event(self, event_id: str) -> Optional[AuditEvent]:
        """Retrieve and decrypt audit event"""
        try:
            event_key = f"audit:event:{event_id}"
            # Handle both sync and async Redis clients
            hgetall_result = self.redis.hgetall(event_key)
            if hasattr(hgetall_result, '__await__'):
                encrypted_event = await hgetall_result
            else:
                encrypted_event = await hgetall_result
            
            if not encrypted_event:
                return None
            
            # Convert bytes to strings if needed
            if encrypted_event and isinstance(next(iter(encrypted_event.keys()), None), bytes):
                encrypted_event = {k.decode(): v.decode() for k, v in encrypted_event.items()}
            
            # Decrypt event
            event_data = self.integrity_manager.decrypt_event(encrypted_event)
            if not event_data:
                return None
            
            # Parse event
            event_dict = json.loads(event_data)
            return AuditEvent.from_dict(event_dict)
            
        except Exception as e:
            logger.error(f"Failed to retrieve audit event {event_id}: {e}")
            return None
    
    async def query_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """Query audit events with filters"""
        try:
            event_ids: Set[str] = set()
            
            # Query by time range
            if start_time or end_time:
                start_ts = start_time.timestamp() if start_time else 0
                end_ts = end_time.timestamp() if end_time else float('inf')
                
                # Get date range
                current_date = start_time.date() if start_time else datetime.now().date()
                end_date = end_time.date() if end_time else datetime.now().date()
                
                while current_date <= end_date:
                    time_key = f"audit:time:{current_date.strftime('%Y-%m-%d')}"
                    day_events = await self.redis.zrangebyscore(
                        time_key, start_ts, end_ts, withscores=False
                    )
                    event_ids.update(e.decode() for e in day_events)
                    current_date = current_date.replace(day=current_date.day + 1)
            
            # Query by user
            if user_id:
                user_key = f"audit:user:{user_id}"
                start_ts = start_time.timestamp() if start_time else 0
                end_ts = end_time.timestamp() if end_time else float('inf')
                user_events = await self.redis.zrangebyscore(
                    user_key, start_ts, end_ts, withscores=False
                )
                user_event_ids = set(e.decode() for e in user_events)
                
                if event_ids:
                    event_ids &= user_event_ids
                else:
                    event_ids = user_event_ids
            
            # Query by event type
            if event_type:
                type_key = f"audit:type:{event_type.value}"
                start_ts = start_time.timestamp() if start_time else 0
                end_ts = end_time.timestamp() if end_time else float('inf')
                type_events = await self.redis.zrangebyscore(
                    type_key, start_ts, end_ts, withscores=False
                )
                type_event_ids = set(e.decode() for e in type_events)
                
                if event_ids:
                    event_ids &= type_event_ids
                else:
                    event_ids = type_event_ids
            
            # If no filters, get recent events
            if not event_ids and not any([start_time, end_time, user_id, event_type]):
                today_key = f"audit:time:{datetime.now().strftime('%Y-%m-%d')}"
                recent_events = await self.redis.zrevrange(today_key, 0, limit - 1)
                event_ids = set(e.decode() for e in recent_events)
            
            # Retrieve and decrypt events
            events = []
            for event_id in list(event_ids)[:limit]:
                event = await self.get_event(event_id)
                if event:
                    events.append(event)
            
            # Sort by timestamp
            events.sort(key=lambda e: e.timestamp, reverse=True)
            return events
            
        except Exception as e:
            logger.error(f"Failed to query audit events: {e}")
            return []
    
    async def verify_chain_integrity(self, start_event_id: str = None) -> bool:
        """Verify the integrity of the audit log chain"""
        try:
            # Get all events in chronological order
            events = await self.query_events(limit=10000)  # Adjust as needed
            events.sort(key=lambda e: e.timestamp)
            
            previous_hash = "0" * 64  # Genesis hash
            
            for event in events:
                # Get encrypted event data
                event_key = f"audit:event:{event.event_id}"
                # Handle both sync and async Redis clients
                hgetall_result = self.redis.hgetall(event_key)
                if hasattr(hgetall_result, '__await__'):
                    encrypted_event_raw = await hgetall_result
                else:
                    encrypted_event_raw = await hgetall_result
                
                if not encrypted_event_raw:
                    logger.error(f"Event {event.event_id} not found in storage")
                    return False
                
                # Convert bytes to strings if needed
                try:
                    if encrypted_event_raw and isinstance(next(iter(encrypted_event_raw.keys()), None), bytes):
                        encrypted_event = {k.decode(): v.decode() for k, v in encrypted_event_raw.items()}
                    else:
                        encrypted_event = encrypted_event_raw
                except (StopIteration, AttributeError):
                    # Handle empty dict or other edge cases
                    encrypted_event = encrypted_event_raw
                
                # Verify chain hash
                stored_previous_hash = encrypted_event.get('previous_hash')
                if stored_previous_hash != previous_hash:
                    logger.error(f"Chain integrity violation at event {event.event_id}")
                    return False
                
                # Decrypt and verify event data
                event_data = self.integrity_manager.decrypt_event(encrypted_event)
                if not event_data:
                    logger.error(f"Failed to decrypt event {event.event_id}")
                    return False
                
                # Verify chain hash
                expected_hash = self.integrity_manager.create_chain_hash(previous_hash, event_data)
                stored_hash = encrypted_event.get('chain_hash')
                
                if expected_hash != stored_hash:
                    logger.error(f"Hash mismatch at event {event.event_id}")
                    return False
                
                previous_hash = stored_hash
            
            logger.info("Audit log chain integrity verified")
            return True
            
        except Exception as e:
            logger.error(f"Chain integrity verification failed: {e}")
            return False
    
    async def create_authentication_event(
        self,
        user_id: str,
        result: str,
        ip_address: str,
        user_agent: str = None,
        details: Dict[str, Any] = None
    ) -> AuditEvent:
        """Create authentication audit event"""
        event = AuditEvent(
            event_id=secrets.token_urlsafe(16),
            event_type=AuditEventType.AUTHENTICATION,
            level=AuditLevel.INFO if result == "success" else AuditLevel.WARNING,
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            session_id=None,
            ip_address=ip_address,
            user_agent=user_agent,
            resource_type="authentication",
            resource_id=None,
            action="login",
            details=details or {},
            result=result
        )
        
        await self.log_event(event)
        return event
    
    async def create_authorization_event(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        result: str,
        session_id: str = None,
        details: Dict[str, Any] = None
    ) -> AuditEvent:
        """Create authorization audit event"""
        event = AuditEvent(
            event_id=secrets.token_urlsafe(16),
            event_type=AuditEventType.AUTHORIZATION,
            level=AuditLevel.INFO if result == "success" else AuditLevel.WARNING,
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            session_id=session_id,
            ip_address=None,
            user_agent=None,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            details=details or {},
            result=result
        )
        
        await self.log_event(event)
        return event
    
    async def create_data_access_event(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        session_id: str = None,
        ip_address: str = None,
        details: Dict[str, Any] = None
    ) -> AuditEvent:
        """Create data access audit event"""
        event = AuditEvent(
            event_id=secrets.token_urlsafe(16),
            event_type=AuditEventType.DATA_ACCESS,
            level=AuditLevel.INFO,
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=None,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            details=details or {},
            result="success"
        )
        
        await self.log_event(event)
        return event
    
    async def cleanup_expired_events(self):
        """Clean up expired audit events"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            
            # This would be implemented based on your specific cleanup requirements
            # For now, Redis TTL handles the cleanup automatically
            
            logger.info("Audit log cleanup completed")
            
        except Exception as e:
            logger.error(f"Audit log cleanup failed: {e}")
