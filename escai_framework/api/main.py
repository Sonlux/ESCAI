"""
Main FastAPI application for ESCAI Framework API.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import uvicorn

from .auth import AuthManager, get_current_user, User
from .auth_endpoints import auth_router
from .monitoring import monitoring_router
from .analysis import analysis_router
from .websocket import websocket_router
from .middleware import ErrorHandlerMiddleware, RequestValidationMiddleware
from ..storage.database import DatabaseManager
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting ESCAI Framework API")
    
    # Initialize database connections
    db_manager = DatabaseManager()
    await db_manager.initialize()
    
    # Store in app state
    app.state.db_manager = db_manager
    app.state.auth_manager = AuthManager()
    
    yield
    
    # Cleanup
    logger.info("Shutting down ESCAI Framework API")
    await db_manager.close()

# Create FastAPI app
app = FastAPI(
    title="ESCAI Framework API",
    description="Epistemic State and Causal Analysis Intelligence Framework",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(RequestValidationMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Add rate limiting error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Include routers
app.include_router(auth_router, prefix="/api/v1/auth", tags=["authentication"])
app.include_router(monitoring_router, prefix="/api/v1/monitor", tags=["monitoring"])
app.include_router(analysis_router, prefix="/api/v1", tags=["analysis"])
app.include_router(websocket_router, prefix="/ws", tags=["websocket"])

@app.get("/")
@limiter.limit("10/minute")
async def root(request: Request):
    """Root endpoint with API information."""
    return {
        "name": "ESCAI Framework API",
        "version": "1.0.0",
        "description": "Epistemic State and Causal Analysis Intelligence Framework",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
@limiter.limit("30/minute")
async def health_check(request: Request):
    """Health check endpoint."""
    try:
        # Check database connectivity
        db_manager = app.state.db_manager
        health_status = await db_manager.health_check()
        
        return {
            "status": "healthy" if health_status["overall"] else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": health_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )

if __name__ == "__main__":
    uvicorn.run(
        "escai_framework.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )