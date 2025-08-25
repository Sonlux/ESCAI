"""
Monitoring endpoints for ESCAI Framework API.
"""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status, Depends, Request
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from .auth import get_current_user, require_developer, User
from ..models.epistemic_state import EpistemicState
from ..instrumentation.base_instrumentor import MonitoringConfig, MonitoringSummary
from ..instrumentation.langchain_instrumentor import LangChainInstrumentor
from ..instrumentation.autogen_instrumentor import AutoGenInstrumentor
from ..instrumentation.crewai_instrumentor import CrewAIInstrumentor
from ..instrumentation.openai_instrumentor import OpenAIInstrumentor
from ..utils.logging import get_logger

logger = get_logger(__name__)
limiter = Limiter(key_func=get_remote_address)

# Router
monitoring_router = APIRouter()

# Request/Response models
class StartMonitoringRequest(BaseModel):
    """Request model for starting monitoring."""
    agent_id: str = Field(..., description="Unique identifier for the agent")
    framework: str = Field(..., description="Agent framework (langchain, autogen, crewai, openai)")
    config: Dict = Field(default_factory=dict, description="Framework-specific configuration")
    monitoring_config: Optional[Dict] = Field(default_factory=dict, description="Monitoring configuration")

class StartMonitoringResponse(BaseModel):
    """Response model for starting monitoring."""
    session_id: str
    agent_id: str
    framework: str
    status: str
    started_at: datetime
    message: str

class MonitoringStatusResponse(BaseModel):
    """Response model for monitoring status."""
    session_id: str
    agent_id: str
    framework: str
    status: str
    started_at: datetime
    last_activity: Optional[datetime]
    events_captured: int
    performance_overhead: float
    error_count: int

class StopMonitoringResponse(BaseModel):
    """Response model for stopping monitoring."""
    session_id: str
    agent_id: str
    status: str
    stopped_at: datetime
    summary: Dict
    message: str

# Active monitoring sessions
active_sessions: Dict[str, Dict] = {}

# Framework instrumentors (lazy initialization)
instrumentors = {}

def get_instrumentor(framework: str):
    """Get instrumentor for framework with lazy initialization."""
    if framework not in instrumentors:
        try:
            if framework == "langchain":
                instrumentors[framework] = LangChainInstrumentor()
            elif framework == "autogen":
                instrumentors[framework] = AutoGenInstrumentor()
            elif framework == "crewai":
                instrumentors[framework] = CrewAIInstrumentor()
            elif framework == "openai":
                instrumentors[framework] = OpenAIInstrumentor()
            else:
                raise ValueError(f"Unsupported framework: {framework}")
        except Exception as e:
            logger.error(f"Failed to initialize {framework} instrumentor: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Framework {framework} is not available: {str(e)}"
            )
    
    return instrumentors[framework]

@monitoring_router.post("/start", response_model=StartMonitoringResponse)
@limiter.limit("10/minute")
async def start_monitoring(
    request: Request,
    monitoring_request: StartMonitoringRequest,
    current_user: User = Depends(require_developer())
):
    """Start monitoring an agent."""
    try:
        # Validate and get instrumentor
        try:
            instrumentor = get_instrumentor(monitoring_request.framework)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported framework: {monitoring_request.framework}"
            )
        
        # Check if agent is already being monitored
        for session_id, session in active_sessions.items():
            if session["agent_id"] == monitoring_request.agent_id and session["status"] == "active":
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Agent {monitoring_request.agent_id} is already being monitored"
                )
        
        # Generate session ID
        session_id = str(uuid4())
        
        # Instrumentor already obtained above
        
        # Create monitoring configuration
        config = MonitoringConfig(
            agent_id=monitoring_request.agent_id,
            framework=monitoring_request.framework,
            capture_epistemic_states=monitoring_request.monitoring_config.get("capture_epistemic_states", True),
            capture_behavioral_patterns=monitoring_request.monitoring_config.get("capture_behavioral_patterns", True),
            capture_performance_metrics=monitoring_request.monitoring_config.get("capture_performance_metrics", True),
            max_events_per_second=monitoring_request.monitoring_config.get("max_events_per_second", 100),
            buffer_size=monitoring_request.monitoring_config.get("buffer_size", 1000)
        )
        
        # Start monitoring
        await instrumentor.start_monitoring(monitoring_request.agent_id, config.dict())
        
        # Store session info
        session_info = {
            "session_id": session_id,
            "agent_id": monitoring_request.agent_id,
            "framework": monitoring_request.framework,
            "status": "active",
            "started_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "events_captured": 0,
            "performance_overhead": 0.0,
            "error_count": 0,
            "user_id": current_user.user_id,
            "instrumentor": instrumentor
        }
        
        active_sessions[session_id] = session_info
        
        logger.info(f"Started monitoring session {session_id} for agent {monitoring_request.agent_id}")
        
        return StartMonitoringResponse(
            session_id=session_id,
            agent_id=monitoring_request.agent_id,
            framework=monitoring_request.framework,
            status="active",
            started_at=session_info["started_at"],
            message="Monitoring started successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start monitoring: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start monitoring"
        )

@monitoring_router.get("/{session_id}/status", response_model=MonitoringStatusResponse)
@limiter.limit("30/minute")
async def get_monitoring_status(
    request: Request,
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get monitoring session status."""
    try:
        # Check if session exists
        if session_id not in active_sessions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Monitoring session {session_id} not found"
            )
        
        session = active_sessions[session_id]
        
        # Check permissions (users can only view their own sessions unless admin)
        if current_user.user_id != session["user_id"] and "admin" not in current_user.roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to monitoring session"
            )
        
        # Get current stats from instrumentor
        instrumentor = session["instrumentor"]
        stats = await instrumentor.get_monitoring_stats(session_id)
        
        # Update session info
        session["events_captured"] = stats.get("events_captured", 0)
        session["performance_overhead"] = stats.get("performance_overhead", 0.0)
        session["error_count"] = stats.get("error_count", 0)
        session["last_activity"] = stats.get("last_activity", session["last_activity"])
        
        return MonitoringStatusResponse(
            session_id=session_id,
            agent_id=session["agent_id"],
            framework=session["framework"],
            status=session["status"],
            started_at=session["started_at"],
            last_activity=session["last_activity"],
            events_captured=session["events_captured"],
            performance_overhead=session["performance_overhead"],
            error_count=session["error_count"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get monitoring status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get monitoring status"
        )

@monitoring_router.post("/{session_id}/stop", response_model=StopMonitoringResponse)
@limiter.limit("10/minute")
async def stop_monitoring(
    request: Request,
    session_id: str,
    current_user: User = Depends(require_developer())
):
    """Stop monitoring session."""
    try:
        # Check if session exists
        if session_id not in active_sessions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Monitoring session {session_id} not found"
            )
        
        session = active_sessions[session_id]
        
        # Check permissions
        if current_user.user_id != session["user_id"] and "admin" not in current_user.roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to monitoring session"
            )
        
        # Stop monitoring
        instrumentor = session["instrumentor"]
        summary = await instrumentor.stop_monitoring(session_id)
        
        # Update session status
        session["status"] = "stopped"
        session["stopped_at"] = datetime.utcnow()
        
        logger.info(f"Stopped monitoring session {session_id}")
        
        return StopMonitoringResponse(
            session_id=session_id,
            agent_id=session["agent_id"],
            status="stopped",
            stopped_at=session["stopped_at"],
            summary=summary.dict() if summary else {},
            message="Monitoring stopped successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop monitoring: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to stop monitoring"
        )

@monitoring_router.get("/sessions", response_model=List[MonitoringStatusResponse])
@limiter.limit("20/minute")
async def list_monitoring_sessions(
    request: Request,
    status_filter: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """List monitoring sessions."""
    try:
        sessions = []
        
        for session_id, session in active_sessions.items():
            # Check permissions
            if current_user.user_id != session["user_id"] and "admin" not in current_user.roles:
                continue
            
            # Apply status filter
            if status_filter and session["status"] != status_filter:
                continue
            
            # Get current stats
            try:
                instrumentor = session["instrumentor"]
                stats = await instrumentor.get_monitoring_stats(session_id)
                
                sessions.append(MonitoringStatusResponse(
                    session_id=session_id,
                    agent_id=session["agent_id"],
                    framework=session["framework"],
                    status=session["status"],
                    started_at=session["started_at"],
                    last_activity=stats.get("last_activity", session["last_activity"]),
                    events_captured=stats.get("events_captured", 0),
                    performance_overhead=stats.get("performance_overhead", 0.0),
                    error_count=stats.get("error_count", 0)
                ))
            except Exception as e:
                logger.warning(f"Failed to get stats for session {session_id}: {e}")
                # Return basic info without stats
                sessions.append(MonitoringStatusResponse(
                    session_id=session_id,
                    agent_id=session["agent_id"],
                    framework=session["framework"],
                    status=session["status"],
                    started_at=session["started_at"],
                    last_activity=session["last_activity"],
                    events_captured=0,
                    performance_overhead=0.0,
                    error_count=0
                ))
        
        return sessions
        
    except Exception as e:
        logger.error(f"Failed to list monitoring sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list monitoring sessions"
        )

@monitoring_router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit("10/minute")
async def delete_monitoring_session(
    request: Request,
    session_id: str,
    current_user: User = Depends(require_developer())
):
    """Delete monitoring session."""
    try:
        # Check if session exists
        if session_id not in active_sessions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Monitoring session {session_id} not found"
            )
        
        session = active_sessions[session_id]
        
        # Check permissions
        if current_user.user_id != session["user_id"] and "admin" not in current_user.roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to monitoring session"
            )
        
        # Stop monitoring if still active
        if session["status"] == "active":
            instrumentor = session["instrumentor"]
            await instrumentor.stop_monitoring(session_id)
        
        # Remove session
        del active_sessions[session_id]
        
        logger.info(f"Deleted monitoring session {session_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete monitoring session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete monitoring session"
        )