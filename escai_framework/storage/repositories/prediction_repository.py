"""
Repository for PredictionRecord model operations.
"""

from typing import List, Optional, cast
from datetime import datetime, timedelta
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc, asc, func

from .base_repository import BaseRepository
from ..models import PredictionRecord


class PredictionRepository(BaseRepository[PredictionRecord]):
    """Repository for PredictionRecord operations."""
    
    def __init__(self):
        super().__init__(PredictionRecord)
    
    async def get_by_prediction_id(
        self,
        session: AsyncSession,
        prediction_id: str
    ) -> Optional[PredictionRecord]:
        """Get prediction by prediction_id."""
        result = await session.execute(
            select(PredictionRecord)
            .where(PredictionRecord.prediction_id == prediction_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_agent(
        self,
        session: AsyncSession,
        agent_id: UUID,
        prediction_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[PredictionRecord]:
        """Get predictions for an agent."""
        query = select(PredictionRecord).where(
            PredictionRecord.agent_id == agent_id
        )
        
        if prediction_type:
            query = query.where(PredictionRecord.prediction_type == prediction_type)
        
        query = query.order_by(desc(PredictionRecord.predicted_at))
        
        if limit:
            query = query.limit(limit)
        
        result = await session.execute(query)
        return list(result.scalars().all())
    
    async def get_by_session(
        self,
        session: AsyncSession,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[PredictionRecord]:
        """Get predictions for a monitoring session."""
        query = select(PredictionRecord).where(
            PredictionRecord.session_id == session_id
        ).order_by(desc(PredictionRecord.predicted_at))
        
        if limit:
            query = query.limit(limit)
        
        result = await session.execute(query)
        return list(result.scalars().all())
    
    async def get_recent_predictions(
        self,
        session: AsyncSession,
        agent_id: UUID,
        hours: int = 24,
        limit: Optional[int] = None
    ) -> List[PredictionRecord]:
        """Get recent predictions for an agent."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        query = select(PredictionRecord).where(
            and_(
                PredictionRecord.agent_id == agent_id,
                PredictionRecord.predicted_at >= cutoff_time
            )
        ).order_by(desc(PredictionRecord.predicted_at))
        
        if limit:
            query = query.limit(limit)
        
        result = await session.execute(query)
        return list(result.scalars().all())
    
    async def get_high_confidence_predictions(
        self,
        session: AsyncSession,
        agent_id: UUID,
        min_confidence: float = 0.8,
        limit: Optional[int] = None
    ) -> List[PredictionRecord]:
        """Get high confidence predictions."""
        query = select(PredictionRecord).where(
            and_(
                PredictionRecord.agent_id == agent_id,
                PredictionRecord.confidence_score >= min_confidence
            )
        ).order_by(desc(PredictionRecord.confidence_score))
        
        if limit:
            query = query.limit(limit)
        
        result = await session.execute(query)
        return list(result.scalars().all())
    
    async def get_validated_predictions(
        self,
        session: AsyncSession,
        agent_id: UUID,
        min_accuracy: Optional[float] = None,
        limit: Optional[int] = None
    ) -> List[PredictionRecord]:
        """Get validated predictions."""
        query = select(PredictionRecord).where(
            and_(
                PredictionRecord.agent_id == agent_id,
                PredictionRecord.validated_at.isnot(None),
                PredictionRecord.accuracy_score.isnot(None)
            )
        )
        
        if min_accuracy is not None:
            query = query.where(PredictionRecord.accuracy_score >= min_accuracy)
        
        query = query.order_by(desc(PredictionRecord.validated_at))
        
        if limit:
            query = query.limit(limit)
        
        result = await session.execute(query)
        return list(result.scalars().all())
    
    async def validate_prediction(
        self,
        session: AsyncSession,
        prediction_id: str,
        actual_value: dict,
        accuracy_score: float,
        validation_method: str = 'manual'
    ) -> Optional[PredictionRecord]:
        """Validate a prediction with actual results."""
        prediction = await self.get_by_prediction_id(session, prediction_id)
        if prediction:
            prediction.actual_value = cast(dict, actual_value)
            prediction.accuracy_score = cast(float, accuracy_score)
            prediction.validated_at = cast(datetime, datetime.utcnow())
            prediction.validation_method = cast(str, validation_method)
            
            await session.flush()
            await session.refresh(prediction)
            return prediction
        return None
    
    async def get_accuracy_statistics(
        self,
        session: AsyncSession,
        agent_id: UUID,
        prediction_type: Optional[str] = None,
        days: int = 30
    ) -> dict:
        """Get prediction accuracy statistics."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        query = select(
            func.count(PredictionRecord.id).label('total_predictions'),
            func.count(PredictionRecord.accuracy_score).label('validated_predictions'),
            func.avg(PredictionRecord.accuracy_score).label('avg_accuracy'),
            func.min(PredictionRecord.accuracy_score).label('min_accuracy'),
            func.max(PredictionRecord.accuracy_score).label('max_accuracy'),
            func.stddev(PredictionRecord.accuracy_score).label('std_accuracy'),
            func.avg(PredictionRecord.confidence_score).label('avg_confidence')
        ).where(
            and_(
                PredictionRecord.agent_id == agent_id,
                PredictionRecord.predicted_at >= cutoff_date
            )
        )
        
        if prediction_type:
            query = query.where(PredictionRecord.prediction_type == prediction_type)
        
        result = await session.execute(query)
        row = result.first()
        
        return {
            'total_predictions': row.total_predictions,
            'validated_predictions': row.validated_predictions,
            'validation_rate': (
                row.validated_predictions / row.total_predictions 
                if row.total_predictions > 0 else 0.0
            ),
            'avg_accuracy': float(row.avg_accuracy) if row.avg_accuracy else 0.0,
            'min_accuracy': float(row.min_accuracy) if row.min_accuracy else 0.0,
            'max_accuracy': float(row.max_accuracy) if row.max_accuracy else 0.0,
            'std_accuracy': float(row.std_accuracy) if row.std_accuracy else 0.0,
            'avg_confidence': float(row.avg_confidence) if row.avg_confidence else 0.0
        }
    
    async def get_model_performance(
        self,
        session: AsyncSession,
        model_name: str,
        days: int = 30
    ) -> dict:
        """Get performance statistics for a specific model."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        result = await session.execute(
            select(
                func.count(PredictionRecord.id).label('total_predictions'),
                func.count(PredictionRecord.accuracy_score).label('validated_predictions'),
                func.avg(PredictionRecord.accuracy_score).label('avg_accuracy'),
                func.avg(PredictionRecord.confidence_score).label('avg_confidence')
            )
            .where(
                and_(
                    PredictionRecord.model_name == model_name,
                    PredictionRecord.predicted_at >= cutoff_date
                )
            )
        )
        row = result.first()
        
        return {
            'model_name': model_name,
            'total_predictions': row.total_predictions,
            'validated_predictions': row.validated_predictions,
            'avg_accuracy': float(row.avg_accuracy) if row.avg_accuracy else 0.0,
            'avg_confidence': float(row.avg_confidence) if row.avg_confidence else 0.0
        }