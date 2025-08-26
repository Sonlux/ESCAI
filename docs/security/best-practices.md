# ESCAI Framework Security Best Practices

## Deployment Security

### Production Environment Setup

#### 1. Infrastructure Security

**Network Security**:

- Deploy behind a Web Application Firewall (WAF)
- Use private networks for internal communication
- Implement network segmentation
- Enable DDoS protection
- Use load balancers with SSL termination

**Container Security**:

```dockerfile
# Use minimal base images
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r escai && useradd -r -g escai escai

# Set proper file permissions
COPY --chown=escai:escai . /app
USER escai

# Use security scanning
RUN pip install safety && safety check
```

**Kubernetes Security**:

```yaml
apiVersion: v1
kind: Pod
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 2000
  containers:
    - name: escai-api
      securityContext:
        allowPrivilegeEscalation: false
        readOnlyRootFilesystem: true
        capabilities:
          drop:
            - ALL
```

#### 2. Certificate Management

**Production Certificates**:

```bash
# Use Let's Encrypt for automatic certificate management
certbot certonly --dns-cloudflare \
  --dns-cloudflare-credentials ~/.secrets/cloudflare.ini \
  -d api.escai.yourdomain.com

# Or use cert-manager in Kubernetes
kubectl apply -f https://github.com/jetstack/cert-manager/releases/download/v1.12.0/cert-manager.yaml
```

**Certificate Monitoring**:

```python
# Monitor certificate expiration
async def check_certificate_health():
    tls_manager = TLSManager()

    for hostname in ["api.escai.local", "ws.escai.local"]:
        if await tls_manager.needs_renewal(hostname, days_before_expiry=30):
            # Send alert
            await send_alert(f"Certificate for {hostname} expires soon")
```

#### 3. Secret Management

**Use External Secret Management**:

```python
# Use HashiCorp Vault, AWS Secrets Manager, or Azure Key Vault
import hvac

client = hvac.Client(url='https://vault.company.com')
client.token = os.environ['VAULT_TOKEN']

secret = client.secrets.kv.v2.read_secret_version(path='escai/prod')
config.secret_key = secret['data']['data']['secret_key']
```

**Environment Variable Security**:

```bash
# Use encrypted environment files
# .env.prod (encrypted)
ESCAI_SECRET_KEY=$(vault kv get -field=secret_key secret/escai/prod)
ESCAI_DB_PASSWORD=$(vault kv get -field=db_password secret/escai/prod)
```

### Authentication and Authorization

#### 1. Multi-Factor Authentication

**TOTP Implementation**:

```python
import pyotp
from escai_framework.security import AuthManager

class MFAAuthManager(AuthManager):
    async def verify_mfa_token(self, user_id: str, token: str) -> bool:
        # Get user's TOTP secret from secure storage
        user_secret = await self.get_user_totp_secret(user_id)
        totp = pyotp.TOTP(user_secret)

        return totp.verify(token, valid_window=1)

    async def authenticate_user_with_mfa(
        self,
        username: str,
        password: str,
        mfa_token: str,
        ip_address: str
    ) -> Optional[TokenPair]:
        # First verify password
        if not await self._validate_credentials(username, password):
            return None

        # Then verify MFA token
        user_id = await self.get_user_id(username)
        if not await self.verify_mfa_token(user_id, mfa_token):
            return None

        # Create token pair
        return await self.create_token_pair_for_user(user_id)
```

#### 2. Advanced RBAC Patterns

**Time-Based Access Control**:

```python
# Create time-restricted permissions
time_restricted_permission = await rbac_manager.create_custom_permission(
    resource_type=ResourceType.SYSTEM_CONFIG,
    action=Action.UPDATE,
    conditions={
        "time_range": {
            "start": "09:00",
            "end": "17:00"
        },
        "days": ["monday", "tuesday", "wednesday", "thursday", "friday"]
    }
)
```

**IP-Based Access Control**:

```python
# Create IP-restricted permissions
ip_restricted_permission = await rbac_manager.create_custom_permission(
    resource_type=ResourceType.USER_MANAGEMENT,
    action=Action.ADMIN,
    conditions={
        "allowed_ips": ["192.168.1.0/24", "10.0.0.0/8"],
        "require_vpn": True
    }
)
```

#### 3. Session Security

**Secure Session Configuration**:

```python
# Configure secure session settings
session_config = {
    "secure": True,  # HTTPS only
    "httponly": True,  # No JavaScript access
    "samesite": "strict",  # CSRF protection
    "max_age": 3600,  # 1 hour timeout
    "domain": ".escai.yourdomain.com"
}
```

### Data Protection

#### 1. Advanced PII Detection

**Custom PII Patterns**:

```python
# Add industry-specific PII patterns
pii_detector.add_custom_pattern(
    "medical_record_number",
    r'\bMRN[-\s]?\d{6,10}\b',
    PIIType.CUSTOM
)

pii_detector.add_custom_pattern(
    "employee_id",
    r'\bEMP[-\s]?\d{4,8}\b',
    PIIType.CUSTOM
)

# Add context-aware detection
def detect_pii_with_context(text: str, context: Dict[str, Any]) -> List[PIIMatch]:
    matches = pii_detector.detect_pii(text)

    # Adjust sensitivity based on context
    if context.get("data_classification") == "public":
        # Filter out low-confidence matches
        matches = [m for m in matches if m.confidence > 0.8]
    elif context.get("data_classification") == "restricted":
        # Include all matches, even low confidence
        pass

    return matches
```

#### 2. Data Classification

**Implement Data Classification**:

```python
from enum import Enum

class DataClassification(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class ClassificationAwarePIIMasker(PIIMasker):
    def mask_by_classification(
        self,
        data: Dict[str, Any],
        classification: DataClassification
    ) -> Dict[str, Any]:
        if classification == DataClassification.PUBLIC:
            # Aggressive masking for public data
            self.set_masking_rule(PIIType.EMAIL, MaskingRule(
                PIIType.EMAIL,
                replacement_pattern="[EMAIL_REDACTED]"
            ))
        elif classification == DataClassification.RESTRICTED:
            # Hash all PII for restricted data
            for pii_type in PIIType:
                self.set_masking_rule(pii_type, MaskingRule(
                    pii_type,
                    hash_instead=True
                ))

        return self.mask_structured_data(data, pii_detector)
```

#### 3. Encryption at Rest

**Database Encryption**:

```python
# Use database-level encryption
DATABASE_CONFIG = {
    "postgresql": {
        "sslmode": "require",
        "sslcert": "/etc/ssl/certs/client-cert.pem",
        "sslkey": "/etc/ssl/private/client-key.pem",
        "sslrootcert": "/etc/ssl/certs/ca-cert.pem"
    }
}

# Application-level encryption for sensitive fields
from cryptography.fernet import Fernet

class EncryptedField:
    def __init__(self, key: bytes):
        self.cipher = Fernet(key)

    def encrypt(self, value: str) -> str:
        return self.cipher.encrypt(value.encode()).decode()

    def decrypt(self, encrypted_value: str) -> str:
        return self.cipher.decrypt(encrypted_value.encode()).decode()
```

### Monitoring and Alerting

#### 1. Security Event Monitoring

**Real-time Security Monitoring**:

```python
import asyncio
from typing import List

class SecurityMonitor:
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self.alert_rules = []

    async def monitor_security_events(self):
        """Monitor for security events and trigger alerts"""
        while True:
            try:
                # Check for suspicious patterns
                await self.check_failed_logins()
                await self.check_privilege_escalation()
                await self.check_unusual_access_patterns()
                await self.check_pii_exposure()

                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Security monitoring error: {e}")

    async def check_failed_logins(self):
        """Check for brute force attacks"""
        recent_events = await self.audit_logger.query_events(
            event_type=AuditEventType.AUTHENTICATION,
            start_time=datetime.now() - timedelta(minutes=5)
        )

        failed_attempts = {}
        for event in recent_events:
            if event.result == "failure":
                ip = event.ip_address
                failed_attempts[ip] = failed_attempts.get(ip, 0) + 1

        for ip, count in failed_attempts.items():
            if count >= 10:  # 10 failures in 5 minutes
                await self.send_alert(
                    f"Potential brute force attack from {ip}: {count} failed attempts"
                )

    async def check_privilege_escalation(self):
        """Check for privilege escalation attempts"""
        recent_events = await self.audit_logger.query_events(
            event_type=AuditEventType.AUTHORIZATION,
            start_time=datetime.now() - timedelta(minutes=10)
        )

        denied_attempts = {}
        for event in recent_events:
            if event.result == "failure":
                user_id = event.user_id
                denied_attempts[user_id] = denied_attempts.get(user_id, 0) + 1

        for user_id, count in denied_attempts.items():
            if count >= 5:  # 5 denials in 10 minutes
                await self.send_alert(
                    f"Potential privilege escalation attempt by user {user_id}: {count} denials"
                )
```

#### 2. Anomaly Detection

**Behavioral Anomaly Detection**:

```python
from sklearn.ensemble import IsolationForest
import numpy as np

class SecurityAnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=0.1)
        self.is_trained = False

    def extract_features(self, events: List[AuditEvent]) -> np.ndarray:
        """Extract features from audit events"""
        features = []
        for event in events:
            feature_vector = [
                hash(event.user_id) % 1000,  # User ID hash
                event.timestamp.hour,  # Hour of day
                event.timestamp.weekday(),  # Day of week
                len(event.details),  # Detail complexity
                1 if event.result == "success" else 0,  # Success flag
            ]
            features.append(feature_vector)

        return np.array(features)

    async def train_model(self, audit_logger: AuditLogger):
        """Train anomaly detection model on historical data"""
        # Get last 30 days of events
        events = await audit_logger.query_events(
            start_time=datetime.now() - timedelta(days=30),
            limit=10000
        )

        if len(events) < 100:
            return  # Not enough data

        features = self.extract_features(events)
        self.model.fit(features)
        self.is_trained = True

    async def detect_anomalies(self, recent_events: List[AuditEvent]) -> List[AuditEvent]:
        """Detect anomalous events"""
        if not self.is_trained or not recent_events:
            return []

        features = self.extract_features(recent_events)
        anomaly_scores = self.model.decision_function(features)

        # Return events with anomaly scores below threshold
        anomalous_events = []
        for i, score in enumerate(anomaly_scores):
            if score < -0.5:  # Threshold for anomaly
                anomalous_events.append(recent_events[i])

        return anomalous_events
```

### Incident Response

#### 1. Automated Response

**Security Incident Response**:

```python
class SecurityIncidentResponse:
    def __init__(self, auth_manager: AuthManager, rbac_manager: RBACManager):
        self.auth_manager = auth_manager
        self.rbac_manager = rbac_manager

    async def handle_brute_force_attack(self, ip_address: str):
        """Handle brute force attack"""
        # Block IP address
        await self.block_ip_address(ip_address)

        # Notify security team
        await self.send_security_alert(
            f"Brute force attack detected from {ip_address}. IP blocked."
        )

        # Log incident
        await self.log_security_incident("brute_force", {
            "ip_address": ip_address,
            "action_taken": "ip_blocked"
        })

    async def handle_privilege_escalation(self, user_id: str):
        """Handle privilege escalation attempt"""
        # Temporarily suspend user
        await self.suspend_user(user_id)

        # Revoke all active sessions
        await self.auth_manager.revoke_all_user_sessions(user_id)

        # Notify security team
        await self.send_security_alert(
            f"Privilege escalation attempt by user {user_id}. User suspended."
        )

    async def handle_pii_exposure(self, event_details: Dict[str, Any]):
        """Handle PII exposure incident"""
        # Log incident
        await self.log_security_incident("pii_exposure", event_details)

        # Notify data protection officer
        await self.send_dpo_notification(event_details)

        # Trigger data breach response if necessary
        if event_details.get("severity") == "high":
            await self.trigger_data_breach_response(event_details)
```

#### 2. Compliance Reporting

**Automated Compliance Reports**:

```python
class ComplianceReporter:
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger

    async def generate_gdpr_report(self, start_date: datetime, end_date: datetime):
        """Generate GDPR compliance report"""
        events = await self.audit_logger.query_events(
            start_time=start_date,
            end_time=end_date,
            limit=100000
        )

        report = {
            "period": f"{start_date.date()} to {end_date.date()}",
            "data_access_events": 0,
            "data_modification_events": 0,
            "pii_processing_events": 0,
            "user_consent_events": 0,
            "data_deletion_events": 0
        }

        for event in events:
            if event.event_type == AuditEventType.DATA_ACCESS:
                report["data_access_events"] += 1
            elif event.event_type == AuditEventType.DATA_MODIFICATION:
                report["data_modification_events"] += 1

            # Check for PII processing
            if "pii" in event.details:
                report["pii_processing_events"] += 1

        return report

    async def generate_sox_report(self, start_date: datetime, end_date: datetime):
        """Generate SOX compliance report"""
        events = await self.audit_logger.query_events(
            start_time=start_date,
            end_time=end_date,
            event_type=AuditEventType.SYSTEM_CONFIGURATION
        )

        report = {
            "period": f"{start_date.date()} to {end_date.date()}",
            "configuration_changes": len(events),
            "unauthorized_changes": 0,
            "change_approvals": 0
        }

        for event in events:
            if event.result == "failure":
                report["unauthorized_changes"] += 1
            if "approved_by" in event.details:
                report["change_approvals"] += 1

        return report
```

### Performance and Scalability

#### 1. Security Performance Optimization

**Caching Security Decisions**:

```python
import asyncio
from functools import lru_cache
import time

class CachedRBACManager(RBACManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.permission_cache = {}
        self.cache_ttl = 300  # 5 minutes

    async def check_permission_cached(
        self,
        user_id: str,
        resource_type: ResourceType,
        action: Action,
        resource_id: str = None
    ) -> bool:
        cache_key = f"{user_id}:{resource_type.value}:{action.value}:{resource_id}"

        # Check cache
        if cache_key in self.permission_cache:
            cached_result, timestamp = self.permission_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_result

        # Check permission
        result = await self.check_permission(user_id, resource_type, action, resource_id)

        # Cache result
        self.permission_cache[cache_key] = (result, time.time())

        return result
```

#### 2. Distributed Security

**Multi-Instance Security Coordination**:

```python
class DistributedSecurityManager:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.instance_id = secrets.token_hex(8)

    async def coordinate_security_action(self, action: str, data: Dict[str, Any]):
        """Coordinate security actions across instances"""
        message = {
            "action": action,
            "data": data,
            "instance_id": self.instance_id,
            "timestamp": datetime.utcnow().isoformat()
        }

        await self.redis.publish("security_coordination", json.dumps(message))

    async def handle_security_coordination(self):
        """Handle security coordination messages"""
        pubsub = self.redis.pubsub()
        await pubsub.subscribe("security_coordination")

        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    data = json.loads(message["data"])

                    # Ignore messages from this instance
                    if data["instance_id"] == self.instance_id:
                        continue

                    # Handle different actions
                    if data["action"] == "block_ip":
                        await self.block_ip_locally(data["data"]["ip_address"])
                    elif data["action"] == "revoke_token":
                        await self.revoke_token_locally(data["data"]["token_id"])

                except Exception as e:
                    logger.error(f"Security coordination error: {e}")
```

This comprehensive security implementation provides enterprise-grade protection for the ESCAI framework with all the required security features and best practices.
