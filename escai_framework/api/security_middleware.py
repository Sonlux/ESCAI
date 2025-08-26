"""
Security Middleware for ESCAI Framework API

Integrates all security components into FastAPI middleware for
comprehensive request processing and protection.
"""

import asyncio
import time
from typing import Callable, Dict, Any, Optional
from datetime import datetime

from fastapi import Request, Response, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
import redis.asyncio as redis

from ..security import (
    AuthManager,
    RBACManager,
    PIIDetector,
    PIIMasker,
    AuditLogger,
    InputValidator,
    ResourceType,
    Action,
    AuditEventType,
    AuditLevel,
    SensitivityLevel,
    ValidationLevel
)
from ..security.config import get_security_config
from ..utils.logging import get_logger

logger = get_logger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Comprehensive security middleware"""
    
    def __init__(
        self,
        app,
        redis_client: redis.Redis,
        security_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(app)
        self.redis = redis_client
        self.config = security_config or get_security_config()
        
        # Initialize security components
        self.auth_manager = AuthManager(redis_client)
        self.rbac_manager = RBACManager(redis_client)
        self.audit_logger = AuditLogger(redis_client, self.config.secret_key)
        self.input_validator = InputValidator(
            ValidationLevel[self.config.validation.validation_level]
        )
        self.pii_detector = PIIDetector(
            SensitivityLevel[self.config.pii.sensitivity_level]
        )
        self.pii_masker = PIIMasker()
        
        # Security bypass paths (health checks, docs, etc.)
        self.bypass_paths = {
            "/health", "/health/ready", "/health/live", "/metrics",
            "/docs", "/redoc", "/openapi.json", "/favicon.ico"
        }
        
        # Public paths that don't require authentication
        self.public_paths = {
            "/", "/api/v1/auth/login", "/api/v1/auth/register"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Main middleware dispatch method"""
        start_time = time.time()
        
        try:
            # Skip security for bypass paths
            if any(request.url.path.startswith(path) for path in self.bypass_paths):
                return await call_next(request)
            
            # 1. Input validation and sanitization
            await self._validate_and_sanitize_input(request)
            
            # 2. Authentication (skip for public paths)
            user_context = None
            if not any(request.url.path.startswith(path) for path in self.public_paths):
                user_context = await self._authenticate_request(request)
            
            # 3. Authorization (if authenticated)
            if user_context:
                await self._authorize_request(request, user_context)
            
            # 4. PII detection and masking (for request data)
            await self._process_pii_in_request(request)
            
            # Process request
            response = await call_next(request)
            
            # 5. PII detection and masking (for response data)
            response = await self._process_pii_in_response(response)
            
            # 6. Audit logging
            await self._log_request(request, response, user_context, start_time)
            
            return response
            
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            
            # Log security error
            await self._log_security_error(request, str(e), start_time)
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal security error"
            )
    
    async def _validate_and_sanitize_input(self, request: Request):
        """Validate and sanitize request input"""
        try:
            # Get request data
            request_data = {}
            
            # Query parameters
            if request.query_params:
                request_data.update(dict(request.query_params))
            
            # Path parameters
            if hasattr(request, 'path_params') and request.path_params:
                request_data.update(request.path_params)
            
            # JSON body (if present)
            if request.headers.get('content-type', '').startswith('application/json'):
                try:
                    body = await request.body()
                    if body:
                        import json
                        json_data = json.loads(body)
                        if isinstance(json_data, dict):
                            request_data.update(json_data)
                except Exception:
                    pass  # Invalid JSON will be handled by FastAPI
            
            # Validate input
            if request_data:
                validation_result = self.input_validator.validate_data(request_data)
                
                if not validation_result.is_valid:
                    logger.warning(f"Input validation failed: {validation_result.errors}")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={
                            "error": "Input validation failed",
                            "details": validation_result.errors
                        }
                    )
                
                # Store sanitized data for later use
                request.state.sanitized_data = validation_result.sanitized_data
                request.state.validation_warnings = validation_result.warnings
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Input validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input validation failed"
            )
    
    async def _authenticate_request(self, request: Request) -> Dict[str, Any]:
        """Authenticate request and return user context"""
        try:
            # Extract token from Authorization header
            auth_header = request.headers.get('authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Missing or invalid authorization header"
                )
            
            token = auth_header.split(' ')[1]
            
            # Validate token
            claims = await self.auth_manager.token_manager.validate_access_token(token)
            if not claims:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token"
                )
            
            # Check if token is blacklisted
            if await self.auth_manager.token_manager.is_token_blacklisted(token):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )
            
            # Create user context
            user_context = {
                'user_id': claims['sub'],
                'username': claims.get('username'),
                'email': claims.get('email'),
                'roles': claims.get('roles', []),
                'permissions': claims.get('permissions', []),
                'session_id': claims.get('session_id'),
                'ip_address': request.client.host if request.client else None,
                'user_agent': request.headers.get('user-agent')
            }
            
            # Store in request state
            request.state.user_context = user_context
            
            return user_context
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed"
            )
    
    async def _authorize_request(self, request: Request, user_context: Dict[str, Any]):
        """Authorize request based on user permissions"""
        try:
            # Map request path to resource type and action
            resource_type, action, resource_id = self._map_request_to_permission(request)
            
            if resource_type and action:
                # Check permission
                has_permission = await self.rbac_manager.check_permission(
                    user_id=user_context['user_id'],
                    resource_type=resource_type,
                    action=action,
                    resource_id=resource_id,
                    context={
                        'ip_address': user_context.get('ip_address'),
                        'user_agent': user_context.get('user_agent'),
                        'time': datetime.utcnow()
                    }
                )
                
                if not has_permission:
                    # Log authorization failure
                    await self.audit_logger.create_authorization_event(
                        user_id=user_context['user_id'],
                        resource_type=resource_type.value,
                        resource_id=resource_id or "unknown",
                        action=action.value,
                        result="failure",
                        session_id=user_context.get('session_id'),
                        details={
                            'path': str(request.url.path),
                            'method': request.method,
                            'reason': 'insufficient_permissions'
                        }
                    )
                    
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Insufficient permissions"
                    )
                
                # Log successful authorization
                await self.audit_logger.create_authorization_event(
                    user_id=user_context['user_id'],
                    resource_type=resource_type.value,
                    resource_id=resource_id or "unknown",
                    action=action.value,
                    result="success",
                    session_id=user_context.get('session_id')
                )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Authorization error: {e}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Authorization failed"
            )
    
    def _map_request_to_permission(self, request: Request) -> tuple:
        """Map HTTP request to RBAC resource type and action"""
        path = request.url.path
        method = request.method.upper()
        
        # Extract resource ID from path if present
        resource_id = None
        path_parts = path.strip('/').split('/')
        
        # Map paths to resources
        if '/monitor' in path:
            resource_type = ResourceType.MONITORING_SESSION
            if method == 'GET':
                action = Action.READ
            elif method == 'POST':
                action = Action.CREATE
            elif method in ['PUT', 'PATCH']:
                action = Action.UPDATE
            elif method == 'DELETE':
                action = Action.DELETE
            else:
                action = Action.READ
                
            # Extract session ID if present
            if len(path_parts) > 3:
                resource_id = path_parts[3]
                
        elif '/epistemic' in path:
            resource_type = ResourceType.EPISTEMIC_STATE
            action = Action.READ if method == 'GET' else Action.UPDATE
            
        elif '/patterns' in path:
            resource_type = ResourceType.BEHAVIORAL_PATTERN
            action = Action.READ if method == 'GET' else Action.UPDATE
            
        elif '/causal' in path:
            resource_type = ResourceType.CAUSAL_RELATIONSHIP
            action = Action.READ if method == 'GET' else Action.UPDATE
            
        elif '/predictions' in path:
            resource_type = ResourceType.PREDICTION
            action = Action.READ if method == 'GET' else Action.UPDATE
            
        elif '/admin' in path or '/config' in path:
            resource_type = ResourceType.SYSTEM_CONFIG
            action = Action.ADMIN
            
        else:
            # Default to agent resource
            resource_type = ResourceType.AGENT
            action = Action.READ if method == 'GET' else Action.UPDATE
        
        return resource_type, action, resource_id
    
    async def _process_pii_in_request(self, request: Request):
        """Detect and mask PII in request data"""
        try:
            if hasattr(request.state, 'sanitized_data'):
                data = request.state.sanitized_data
                
                # Detect PII
                pii_matches = []
                for field, value in data.items():
                    if isinstance(value, str):
                        matches = self.pii_detector.detect_pii(value, context=field)
                        pii_matches.extend(matches)
                
                if pii_matches and self.config.pii.auto_mask:
                    # Mask PII
                    masked_data = self.pii_masker.mask_structured_data(data, self.pii_detector)
                    request.state.sanitized_data = masked_data
                    request.state.pii_detected = True
                    request.state.pii_summary = self.pii_masker.get_masking_summary(pii_matches)
                else:
                    request.state.pii_detected = False
                    
        except Exception as e:
            logger.error(f"PII processing error: {e}")
            # Don't fail the request for PII processing errors
    
    async def _process_pii_in_response(self, response: Response) -> Response:
        """Detect and mask PII in response data"""
        try:
            # This would require intercepting response body
            # For now, just return the response as-is
            # In a full implementation, you'd need to:
            # 1. Read response body
            # 2. Detect PII
            # 3. Mask PII if found
            # 4. Return modified response
            
            return response
            
        except Exception as e:
            logger.error(f"Response PII processing error: {e}")
            return response
    
    async def _log_request(
        self,
        request: Request,
        response: Response,
        user_context: Optional[Dict[str, Any]],
        start_time: float
    ):
        """Log request for audit purposes"""
        try:
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Determine event type based on request
            if request.url.path.startswith('/api/v1/auth'):
                event_type = AuditEventType.AUTHENTICATION
            elif request.method in ['POST', 'PUT', 'PATCH', 'DELETE']:
                event_type = AuditEventType.DATA_MODIFICATION
            else:
                event_type = AuditEventType.DATA_ACCESS
            
            # Create audit event
            await self.audit_logger.create_data_access_event(
                user_id=user_context['user_id'] if user_context else 'anonymous',
                resource_type='api_endpoint',
                resource_id=request.url.path,
                action=request.method.lower(),
                session_id=user_context.get('session_id') if user_context else None,
                ip_address=request.client.host if request.client else None,
                details={
                    'path': str(request.url.path),
                    'method': request.method,
                    'status_code': response.status_code,
                    'duration_ms': duration_ms,
                    'user_agent': request.headers.get('user-agent'),
                    'pii_detected': getattr(request.state, 'pii_detected', False),
                    'validation_warnings': getattr(request.state, 'validation_warnings', [])
                }
            )
            
        except Exception as e:
            logger.error(f"Audit logging error: {e}")
            # Don't fail the request for audit logging errors
    
    async def _log_security_error(
        self,
        request: Request,
        error_message: str,
        start_time: float
    ):
        """Log security errors"""
        try:
            duration_ms = int((time.time() - start_time) * 1000)
            
            await self.audit_logger.log_event({
                'event_type': AuditEventType.SECURITY_EVENT,
                'level': AuditLevel.ERROR,
                'timestamp': datetime.utcnow(),
                'user_id': getattr(request.state, 'user_context', {}).get('user_id', 'unknown'),
                'session_id': getattr(request.state, 'user_context', {}).get('session_id'),
                'ip_address': request.client.host if request.client else None,
                'user_agent': request.headers.get('user-agent'),
                'resource_type': 'security_middleware',
                'resource_id': None,
                'action': 'process_request',
                'details': {
                    'error': error_message,
                    'path': str(request.url.path),
                    'method': request.method,
                    'duration_ms': duration_ms
                },
                'result': 'error'
            })
            
        except Exception as e:
            logger.error(f"Security error logging failed: {e}")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware with Redis backend"""
    
    def __init__(self, app, redis_client: redis.Redis, requests_per_minute: int = 60):
        super().__init__(app)
        self.redis = redis_client
        self.requests_per_minute = requests_per_minute
        self.window_size = 60  # 1 minute window
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Rate limit requests by IP address"""
        try:
            client_ip = request.client.host if request.client else "unknown"
            
            # Skip rate limiting for health checks
            if request.url.path.startswith('/health'):
                return await call_next(request)
            
            # Check rate limit
            key = f"rate_limit:{client_ip}"
            current_requests = await self.redis.get(key)
            
            if current_requests is None:
                # First request in window
                await self.redis.setex(key, self.window_size, 1)
            else:
                current_count = int(current_requests)
                if current_count >= self.requests_per_minute:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Rate limit exceeded"
                    )
                
                # Increment counter
                await self.redis.incr(key)
            
            return await call_next(request)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # Don't fail requests for rate limiting errors
            return await call_next(request)