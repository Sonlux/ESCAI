"""
Analysis endpoints for ESCAI Framework API.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, status, Depends, Request, Query
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from .auth import get_current_user, require_researcher, require_viewer, User
from ..models.epistemic_state import EpistemicState
from ..models.behavioral_pattern import BehavioralPattern
from ..models.causal_relationship import CausalRelationship
from ..models.prediction_result import PredictionResult
from ..core.epistemic_extractor import EpistemicExtractor
from ..core.pattern_analyzer import BehavioralAnalyzer
from ..core.causal_engine import CausalEngine
from ..core.performance_predictor import PerformancePredictor
from ..core.explanation_engine import ExplanationEngine
from ..utils.logging import get_logger

logger = get_logger(__name__)
limiter = Limiter(key_func=get_remote_address)

# Router
analysis_router = APIRouter()

# Pagination models
class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Field(1, ge=1, description="Page number")
    size: int = Field(20, ge=1, le=100, description="Page size")

class PaginatedResponse(BaseModel):
    """Paginated response wrapper."""
    items: List[Any]
    total: int
    page: int
    size: int
    pages: int

# Request/Response models
class EpistemicStateQuery(BaseModel):
    """Query parameters for epistemic states."""
    agent_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    confidence_min: Optional[float] = Field(None, ge=0.0, le=1.0)
    confidence_max: Optional[float] = Field(None, ge=0.0, le=1.0)
    uncertainty_min: Optional[float] = Field(None, ge=0.0, le=1.0)
    uncertainty_max: Optional[float] = Field(None, ge=0.0, le=1.0)

class PatternAnalysisQuery(BaseModel):
    """Query parameters for pattern analysis."""
    agent_id: Optional[str] = None
    pattern_type: Optional[str] = None
    success_rate_min: Optional[float] = Field(None, ge=0.0, le=1.0)
    success_rate_max: Optional[float] = Field(None, ge=0.0, le=1.0)
    frequency_min: Optional[int] = Field(None, ge=0)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

class CausalAnalysisRequest(BaseModel):
    """Request for causal analysis."""
    agent_id: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    event_types: Optional[List[str]] = None
    min_confidence: float = Field(0.5, ge=0.0, le=1.0)
    max_relationships: int = Field(100, ge=1, le=1000)

class PredictionRequest(BaseModel):
    """Request for performance prediction."""
    agent_id: str
    current_state: Optional[Dict] = None
    prediction_horizon: int = Field(60, ge=1, le=3600, description="Prediction horizon in seconds")
    include_risk_factors: bool = True
    include_interventions: bool = True

class ExplanationRequest(BaseModel):
    """Request for behavior explanation."""
    agent_id: str
    behavior_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    explanation_type: str = Field("comprehensive", pattern="^(comprehensive|causal|behavioral|predictive)$")
    max_length: int = Field(500, ge=100, le=2000)

# Initialize analysis engines
epistemic_extractor = EpistemicExtractor()
behavioral_analyzer = BehavioralAnalyzer()
causal_engine = CausalEngine()
performance_predictor = PerformancePredictor()
explanation_engine = ExplanationEngine()

@analysis_router.get("/epistemic/{agent_id}/current", response_model=EpistemicState)
@limiter.limit("30/minute")
async def get_current_epistemic_state(
    request: Request,
    agent_id: str,
    current_user: User = Depends(require_viewer())
) -> EpistemicState:
    """Get current epistemic state for an agent."""
    try:
        # Get current epistemic state
        epistemic_state = await epistemic_extractor.get_current_state(agent_id)
        
        if not epistemic_state:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No epistemic state found for agent {agent_id}"
            )
        
        return epistemic_state
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get current epistemic state: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get current epistemic state"
        )

@analysis_router.post("/epistemic/search", response_model=PaginatedResponse)
@limiter.limit("20/minute")
async def search_epistemic_states(
    request: Request,
    query: EpistemicStateQuery,
    pagination: PaginationParams = Depends(),
    current_user: User = Depends(require_viewer())
) -> PaginatedResponse:
    """Search epistemic states with filtering and pagination."""
    try:
        # Build search filters
        filters = {}
        if query.agent_id:
            filters["agent_id"] = query.agent_id
        if query.start_time:
            filters["start_time"] = query.start_time
        if query.end_time:
            filters["end_time"] = query.end_time
        if query.confidence_min is not None:
            filters["confidence_min"] = query.confidence_min
        if query.confidence_max is not None:
            filters["confidence_max"] = query.confidence_max
        if query.uncertainty_min is not None:
            filters["uncertainty_min"] = query.uncertainty_min
        if query.uncertainty_max is not None:
            filters["uncertainty_max"] = query.uncertainty_max
        
        # Search epistemic states
        results = await epistemic_extractor.search_states(
            filters=filters,
            page=pagination.page,
            size=pagination.size
        )
        
        return PaginatedResponse(
            items=results["items"],
            total=results["total"],
            page=pagination.page,
            size=pagination.size,
            pages=(results["total"] + pagination.size - 1) // pagination.size
        )
        
    except Exception as e:
        logger.error(f"Failed to search epistemic states: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search epistemic states"
        )

@analysis_router.post("/patterns/analyze", response_model=PaginatedResponse)
@limiter.limit("15/minute")
async def analyze_behavioral_patterns(
    request: Request,
    query: PatternAnalysisQuery,
    pagination: PaginationParams = Depends(),
    current_user: User = Depends(require_researcher())
) -> PaginatedResponse:
    """Analyze behavioral patterns with filtering."""
    try:
        # Build analysis filters
        filters = {}
        if query.agent_id:
            filters["agent_id"] = query.agent_id
        if query.pattern_type:
            filters["pattern_type"] = query.pattern_type
        if query.success_rate_min is not None:
            filters["success_rate_min"] = query.success_rate_min
        if query.success_rate_max is not None:
            filters["success_rate_max"] = query.success_rate_max
        if query.frequency_min is not None:
            filters["frequency_min"] = query.frequency_min
        if query.start_time:
            filters["start_time"] = query.start_time
        if query.end_time:
            filters["end_time"] = query.end_time
        
        # Analyze patterns
        results = await behavioral_analyzer.analyze_patterns(
            filters=filters,
            page=pagination.page,
            size=pagination.size
        )
        
        return PaginatedResponse(
            items=results["items"],
            total=results["total"],
            page=pagination.page,
            size=pagination.size,
            pages=(results["total"] + pagination.size - 1) // pagination.size
        )
        
    except Exception as e:
        logger.error(f"Failed to analyze behavioral patterns: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze behavioral patterns"
        )

@analysis_router.post("/causal/analyze", response_model=List[CausalRelationship])
@limiter.limit("10/minute")
async def analyze_causal_relationships(
    request: Request,
    analysis_request: CausalAnalysisRequest,
    current_user: User = Depends(require_researcher())
) -> List[CausalRelationship]:
    """Analyze causal relationships in agent behavior."""
    try:
        # Perform causal analysis
        relationships = await causal_engine.discover_relationships(
            agent_id=analysis_request.agent_id,
            start_time=analysis_request.start_time,
            end_time=analysis_request.end_time,
            event_types=analysis_request.event_types,
            min_confidence=analysis_request.min_confidence,
            max_relationships=analysis_request.max_relationships
        )
        
        return relationships
        
    except Exception as e:
        logger.error(f"Failed to analyze causal relationships: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze causal relationships"
        )

@analysis_router.get("/predictions/{agent_id}/current", response_model=PredictionResult)
@limiter.limit("20/minute")
async def get_current_predictions(
    request: Request,
    agent_id: str,
    current_user: User = Depends(require_viewer())
) -> PredictionResult:
    """Get current performance predictions for an agent."""
    try:
        # Get current predictions
        prediction = await performance_predictor.get_current_prediction(agent_id)
        
        if not prediction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No predictions found for agent {agent_id}"
            )
        
        return prediction
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get current predictions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get current predictions"
        )

@analysis_router.post("/predictions/generate", response_model=PredictionResult)
@limiter.limit("10/minute")
async def generate_prediction(
    request: Request,
    prediction_request: PredictionRequest,
    current_user: User = Depends(require_researcher())
) -> PredictionResult:
    """Generate performance prediction for an agent."""
    try:
        # Generate prediction
        prediction = await performance_predictor.predict_performance(
            agent_id=prediction_request.agent_id,
            current_state=prediction_request.current_state,
            prediction_horizon=prediction_request.prediction_horizon,
            include_risk_factors=prediction_request.include_risk_factors,
            include_interventions=prediction_request.include_interventions
        )
        
        return prediction
        
    except Exception as e:
        logger.error(f"Failed to generate prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate prediction"
        )

@analysis_router.post("/explain/behavior", response_model=Dict[str, Any])
@limiter.limit("10/minute")
async def explain_behavior(
    request: Request,
    explanation_request: ExplanationRequest,
    current_user: User = Depends(require_viewer())
) -> Dict[str, Any]:
    """Generate human-readable explanation of agent behavior."""
    try:
        # Generate explanation
        explanation = await explanation_engine.generate_explanation(
            agent_id=explanation_request.agent_id,
            behavior_id=explanation_request.behavior_id,
            start_time=explanation_request.start_time,
            end_time=explanation_request.end_time,
            explanation_type=explanation_request.explanation_type,
            max_length=explanation_request.max_length
        )
        
        return explanation
        
    except Exception as e:
        logger.error(f"Failed to generate explanation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate explanation"
        )

@analysis_router.get("/agents/{agent_id}/summary", response_model=Dict[str, Any])
@limiter.limit("15/minute")
async def get_agent_summary(
    request: Request,
    agent_id: str,
    days: int = Query(7, ge=1, le=90, description="Number of days to include in summary"),
    current_user: User = Depends(require_viewer())
) -> Dict[str, Any]:
    """Get comprehensive summary for an agent."""
    try:
        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        # Get summary data
        summary = await _generate_agent_summary(agent_id, start_time, end_time)
        
        return summary
        
    except Exception as e:
        logger.error(f"Failed to get agent summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get agent summary"
        )

async def _generate_agent_summary(agent_id: str, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
    """Generate comprehensive agent summary."""
    try:
        # Get current epistemic state
        current_state = await epistemic_extractor.get_current_state(agent_id)
        
        # Get recent patterns
        pattern_filters = {
            "agent_id": agent_id,
            "start_time": start_time,
            "end_time": end_time
        }
        patterns = await behavioral_analyzer.analyze_patterns(pattern_filters, page=1, size=10)
        
        # Get recent predictions
        current_prediction = await performance_predictor.get_current_prediction(agent_id)
        
        # Get causal relationships
        relationships = await causal_engine.discover_relationships(
            agent_id=agent_id,
            start_time=start_time,
            end_time=end_time,
            max_relationships=20
        )
        
        # Generate explanation
        explanation = await explanation_engine.generate_explanation(
            agent_id=agent_id,
            start_time=start_time,
            end_time=end_time,
            explanation_type="comprehensive",
            max_length=1000
        )
        
        return {
            "agent_id": agent_id,
            "summary_period": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "days": (end_time - start_time).days
            },
            "current_state": current_state.dict() if current_state else None,
            "behavioral_patterns": {
                "total_patterns": patterns["total"],
                "top_patterns": patterns["items"][:5]
            },
            "current_prediction": current_prediction.dict() if current_prediction else None,
            "causal_relationships": {
                "total_relationships": len(relationships),
                "top_relationships": relationships[:10]
            },
            "explanation": explanation,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to generate agent summary: {e}")
        raise