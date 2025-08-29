"""
Repository for EpistemicStateRecord model operations.
"""

from typing import List, Optional
from datetime import datetime, timedelta
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc, asc, func

from .base_repository import BaseRepository
from ..models import EpistemicStateRecord


class EpistemicStateRepository(BaseRepository[EpistemicStateRecord]):
    """Repository for EpistemicStateRecord operations."""
    
    def __init__(self):
        super().__init__(EpistemicStateRecord)
    
    async def get_latest_by_agent(
        self,
        session: AsyncSession,
        agent_id: UUID
    ) -> Optional[EpistemicStateRecord]:
        """Get the latest epistemic state for an agent."""
        result = await session.execute(
            select(EpistemicStateRecord)
            .where(EpistemicStateRecord.agent_id == agent_id)
            .order_by(desc(EpistemicStateRecord.timestamp))
            .limit(1)
        )
        return result.scalar_one_or_none()
    
    async def get_by_time_range(
        self,
        session: AsyncSession,
        agent_id: UUID,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int] = None
    ) -> List[EpistemicStateRecord]:
        """Get epistemic states within a time range."""
        query = select(EpistemicStateRecord).where(
            and_(
                EpistemicStateRecord.agent_id == agent_id,
                EpistemicStateRecord.timestamp >= start_time,
                EpistemicStateRecord.timestamp <= end_time
            )
        ).order_by(asc(EpistemicStateRecord.timestamp))
        
        if limit:
            query = query.limit(limit)
        
        result = await session.execute(query)
        return list(result.scalars().all())
    
    async def get_by_session(
        self,
        session: AsyncSession,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[EpistemicStateRecord]:
        """Get epistemic states for a monitoring session."""
        query = select(EpistemicStateRecord).where(
            EpistemicStateRecord.session_id == session_id
        ).order_by(asc(EpistemicStateRecord.timestamp))
        
        if limit:
            query = query.limit(limit)
        
        result = await session.execute(query)
        return list(result.scalars().all())
    
    async def get_recent_states(
        self,
        session: AsyncSession,
        agent_id: UUID,
        hours: int = 24,
        limit: Optional[int] = None
    ) -> List[EpistemicStateRecord]:
        """Get recent epistemic states for an agent."""
        start_time = datetime.utcnow() - timedelta(hours=hours)
        return await self.get_by_time_range(
            session, agent_id, start_time, datetime.utcnow(), limit
        )
    
    async def get_confidence_evolution(
        self,
        session: AsyncSession,
        agent_id: UUID,
        start_time: datetime,
        end_time: datetime
    ) -> List[tuple[datetime, float]]:
        """Get confidence level evolution over time."""
        result = await session.execute(
            select(
                EpistemicStateRecord.timestamp,
                EpistemicStateRecord.confidence_level
            )
            .where(
                and_(
                    EpistemicStateRecord.agent_id == agent_id,
                    EpistemicStateRecord.timestamp >= start_time,
                    EpistemicStateRecord.timestamp <= end_time,
                    EpistemicStateRecord.confidence_level.isnot(None)
                )
            )
            .order_by(asc(EpistemicStateRecord.timestamp))
        )
        return [r._tuple() for r in result.all()]
    
    async def get_uncertainty_stats(
        self,
        session: AsyncSession,
        agent_id: UUID,
        start_time: datetime,
        end_time: datetime
    ) -> dict:
        """Get uncertainty statistics for an agent."""
        result = await session.execute(
            select(
                func.avg(EpistemicStateRecord.uncertainty_score).label('avg_uncertainty'),
                func.min(EpistemicStateRecord.uncertainty_score).label('min_uncertainty'),
                func.max(EpistemicStateRecord.uncertainty_score).label('max_uncertainty'),
                func.stddev(EpistemicStateRecord.uncertainty_score).label('std_uncertainty'),
                func.count(EpistemicStateRecord.id).label('count')
            )
            .where(
                and_(
                    EpistemicStateRecord.agent_id == agent_id,
                    EpistemicStateRecord.timestamp >= start_time,
                    EpistemicStateRecord.timestamp <= end_time,
                    EpistemicStateRecord.uncertainty_score.isnot(None)
                )
            )
        )
        row = result.first()
        return {
            'avg_uncertainty': float(row.avg_uncertainty) if row.avg_uncertainty else 0.0,
            'min_uncertainty': float(row.min_uncertainty) if row.min_uncertainty else 0.0,
            'max_uncertainty': float(row.max_uncertainty) if row.max_uncertainty else 0.0,
            'std_uncertainty': float(row.std_uncertainty) if row.std_uncertainty else 0.0,
            'count': row.count
        }
    
    async def cleanup_old_states(
        self,
        session: AsyncSession,
        days_to_keep: int = 90
    ) -> int:
        """Clean up old epistemic states."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        result = await session.execute(
            select(func.count(EpistemicStateRecord.id))
            .where(EpistemicStateRecord.timestamp < cutoff_date)
        )
        count_to_delete = result.scalar()
        
        await session.execute(
            EpistemicStateRecord.__table__.delete()
            .where(EpistemicStateRecord.timestamp < cutoff_date)
        )
        
        return count_to_delete