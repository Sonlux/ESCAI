"""
Middleware for ESCAI Framework API.
"""

import time
import traceback
from typing import Callable, Dict, Any
from uuid import uuid4

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import ValidationError

from ..utils.logging import get_logger

logger = get_logger(__name__)

class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle all requests and catch exceptions."""
        request_id = str(uuid4())
        request.state.request_id = request_id
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Log successful requests
            process_time = time.time() - start_time
            logger.info(
                f"Request {request_id} - {request.method} {request.url.path} - "
                f"Status: {response.status_code} - Time: {process_time:.3f}s"
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except HTTPException as e:
            # Handle FastAPI HTTP exceptions
            process_time = time.time() - start_time
            logger.warning(
                f"Request {request_id} - {request.method} {request.url.path} - "
                f"HTTPException: {e.status_code} - {e.detail} - Time: {process_time:.3f}s"
            )
            
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": {
                        "type": "HTTPException",
                        "message": e.detail,
                        "status_code": e.status_code,
                        "request_id": request_id,
                        "timestamp": time.time()
                    }
                },
                headers={
                    "X-Request-ID": request_id,
                    "X-Process-Time": str(process_time)
                }
            )
            
        except ValidationError as e:
            # Handle Pydantic validation errors
            process_time = time.time() - start_time
            logger.warning(
                f"Request {request_id} - {request.method} {request.url.path} - "
                f"ValidationError: {e} - Time: {process_time:.3f}s"
            )
            
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "error": {
                        "type": "ValidationError",
                        "message": "Request validation failed",
                        "details": e.errors(),
                        "request_id": request_id,
                        "timestamp": time.time()
                    }
                },
                headers={
                    "X-Request-ID": request_id,
                    "X-Process-Time": str(process_time)
                }
            )
            
        except Exception as e:
            # Handle unexpected exceptions
            process_time = time.time() - start_time
            error_trace = traceback.format_exc()
            
            logger.error(
                f"Request {request_id} - {request.method} {request.url.path} - "
                f"Unexpected error: {str(e)} - Time: {process_time:.3f}s\n{error_trace}"
            )
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": {
                        "type": "InternalServerError",
                        "message": "An unexpected error occurred",
                        "request_id": request_id,
                        "timestamp": time.time()
                    }
                },
                headers={
                    "X-Request-ID": request_id,
                    "X-Process-Time": str(process_time)
                }
            )

class RequestValidationMiddleware(BaseHTTPMiddleware):
    """Request validation and sanitization middleware."""
    
    def __init__(self, app, max_request_size: int = 10 * 1024 * 1024):  # 10MB default
        super().__init__(app)
        self.max_request_size = max_request_size
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Validate and sanitize requests."""
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        try:
            # Check request size
            content_length = request.headers.get('content-length')
            if content_length and int(content_length) > self.max_request_size:
                logger.warning(
                    f"Request {request_id} - Request too large: {content_length} bytes"
                )
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={
                        "error": {
                            "type": "RequestTooLarge",
                            "message": f"Request size exceeds maximum allowed size of {self.max_request_size} bytes",
                            "request_id": request_id,
                            "timestamp": time.time()
                        }
                    }
                )
            
            # Validate content type for POST/PUT requests
            if request.method in ["POST", "PUT", "PATCH"]:
                content_type = request.headers.get('content-type', '')
                if content_type and not self._is_valid_content_type(content_type):
                    logger.warning(
                        f"Request {request_id} - Invalid content type: {content_type}"
                    )
                    return JSONResponse(
                        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                        content={
                            "error": {
                                "type": "UnsupportedMediaType",
                                "message": f"Unsupported content type: {content_type}",
                                "request_id": request_id,
                                "timestamp": time.time()
                            }
                        }
                    )
            
            # Add security headers
            response = await call_next(request)
            
            # Add security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            
            return response
            
        except Exception as e:
            logger.error(f"Request validation middleware error: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": {
                        "type": "MiddlewareError",
                        "message": "Request validation failed",
                        "request_id": request_id,
                        "timestamp": time.time()
                    }
                }
            )
    
    def _is_valid_content_type(self, content_type: str) -> bool:
        """Check if content type is valid."""
        valid_types = [
            'application/json',
            'application/x-www-form-urlencoded',
            'multipart/form-data',
            'text/plain'
        ]
        
        # Extract main content type (ignore charset, boundary, etc.)
        main_type = content_type.split(';')[0].strip().lower()
        return main_type in valid_types

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers."""
        response = await call_next(request)
        
        # Security headers
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests for monitoring and debugging."""
    
    def __init__(self, app, log_body: bool = False):
        super().__init__(app)
        self.log_body = log_body
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request details."""
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        # Log request
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get('user-agent', 'unknown')
        
        logger.info(
            f"Request {request_id} - {request.method} {request.url} - "
            f"IP: {client_ip} - User-Agent: {user_agent}"
        )
        
        # Log request body if enabled (be careful with sensitive data)
        if self.log_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    # Truncate large bodies
                    body_str = body.decode('utf-8')[:1000]
                    logger.debug(f"Request {request_id} - Body: {body_str}")
            except Exception as e:
                logger.warning(f"Request {request_id} - Failed to log body: {e}")
        
        response = await call_next(request)
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address considering proxies."""
        # Check for forwarded headers (in order of preference)
        forwarded_headers = [
            'x-forwarded-for',
            'x-real-ip',
            'x-client-ip',
            'cf-connecting-ip'  # Cloudflare
        ]
        
        for header in forwarded_headers:
            if header in request.headers:
                # Take the first IP if there are multiple
                ip = request.headers[header].split(',')[0].strip()
                if ip:
                    return ip
        
        # Fall back to direct client IP
        return request.client.host if request.client else 'unknown'