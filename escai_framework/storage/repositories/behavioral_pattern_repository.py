"""
Repository for BehavioralPatternRecord model operations.
"""

from typing import List, Optional, cast
from datetime import datetime, timedelta
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc, asc, func

from .base_repository import BaseRepository
from ..models import BehavioralPatternRecord


class BehavioralPatternRepository(BaseRepository[BehavioralPatternRecord]):
    """Repository for BehavioralPatternRecord operations."""
    
    def __init__(self):
        super().__init__(BehavioralPatternRecord)
    
    async def get_by_pattern_id(
        self,
        session: AsyncSession,
        pattern_id: str
    ) -> Optional[BehavioralPatternRecord]:
        """Get pattern by pattern_id."""
        result = await session.execute(
            select(BehavioralPatternRecord)
            .where(BehavioralPatternRecord.pattern_id == pattern_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_agent(
        self,
        session: AsyncSession,
        agent_id: UUID,
        pattern_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[BehavioralPatternRecord]:
        """Get patterns for an agent."""
        query = select(BehavioralPatternRecord).where(
            BehavioralPatternRecord.agent_id == agent_id
        )
        
        if pattern_type:
            query = query.where(BehavioralPatternRecord.pattern_type == pattern_type)
        
        query = query.order_by(desc(BehavioralPatternRecord.discovered_at))
        
        if limit:
            query = query.limit(limit)
        
        result = await session.execute(query)
        return list(result.scalars().all())
    
    async def get_high_success_patterns(
        self,
        session: AsyncSession,
        agent_id: UUID,
        min_success_rate: float = 0.8,
        min_frequency: int = 5
    ) -> List[BehavioralPatternRecord]:
        """Get patterns with high success rates."""
        result = await session.execute(
            select(BehavioralPatternRecord)
            .where(
                and_(
                    BehavioralPatternRecord.agent_id == agent_id,
                    BehavioralPatternRecord.success_rate >= min_success_rate,
                    BehavioralPatternRecord.frequency >= min_frequency
                )
            )
            .order_by(desc(BehavioralPatternRecord.success_rate))
        )
        return list(result.scalars().all())
    
    async def get_frequent_patterns(
        self,
        session: AsyncSession,
        agent_id: UUID,
        min_frequency: int = 10,
        limit: Optional[int] = None
    ) -> List[BehavioralPatternRecord]:
        """Get frequently occurring patterns."""
        query = select(BehavioralPatternRecord).where(
            and_(
                BehavioralPatternRecord.agent_id == agent_id,
                BehavioralPatternRecord.frequency >= min_frequency
            )
        ).order_by(desc(BehavioralPatternRecord.frequency))
        
        if limit:
            query = query.limit(limit)
        
        result = await session.execute(query)
        return list(result.scalars().all())
    
    async def get_recent_patterns(
        self,
        session: AsyncSession,
        agent_id: UUID,
        days: int = 7
    ) -> List[BehavioralPatternRecord]:
        """Get recently discovered patterns."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        result = await session.execute(
            select(BehavioralPatternRecord)
            .where(
                and_(
                    BehavioralPatternRecord.agent_id == agent_id,
                    BehavioralPatternRecord.discovered_at >= cutoff_date
                )
            )
            .order_by(desc(BehavioralPatternRecord.discovered_at))
        )
        return list(result.scalars().all())
    
    async def update_pattern_stats(
        self,
        session: AsyncSession,
        pattern_id: str,
        frequency_increment: int = 1,
        success_rate: Optional[float] = None,
        last_observed: Optional[datetime] = None
    ) -> Optional[BehavioralPatternRecord]:
        """Update pattern statistics."""
        pattern = await self.get_by_pattern_id(session, pattern_id)
        if pattern:
            pattern.frequency = cast(int, pattern.frequency + frequency_increment)
            
            if success_rate is not None:
                pattern.success_rate = cast(float, success_rate)
            
            if last_observed:
                pattern.last_observed = cast(datetime, last_observed)
            else:
                pattern.last_observed = cast(datetime, datetime.utcnow())
            
            await session.flush()
            await session.refresh(pattern)
            return pattern
        return None
    
    async def get_pattern_statistics(
        self,
        session: AsyncSession,
        agent_id: UUID
    ) -> dict:
        """Get pattern statistics for an agent."""
        result = await session.execute(
            select(
                func.count(BehavioralPatternRecord.id).label('total_patterns'),
                func.avg(BehavioralPatternRecord.success_rate).label('avg_success_rate'),
                func.avg(BehavioralPatternRecord.frequency).label('avg_frequency'),
                func.max(BehavioralPatternRecord.frequency).label('max_frequency'),
                func.count(
                    BehavioralPatternRecord.id
                ).filter(
                    BehavioralPatternRecord.success_rate >= 0.8
                ).label('high_success_patterns')
            )
            .where(BehavioralPatternRecord.agent_id == agent_id)
        )
        row = result.first()
        return {
            'total_patterns': row.total_patterns,
            'avg_success_rate': float(row.avg_success_rate) if row.avg_success_rate else 0.0,
            'avg_frequency': float(row.avg_frequency) if row.avg_frequency else 0.0,
            'max_frequency': row.max_frequency or 0,
            'high_success_patterns': row.high_success_patterns
        }
    
    async def get_patterns_by_type(
        self,
        session: AsyncSession,
        pattern_type: str,
        limit: Optional[int] = None
    ) -> List[BehavioralPatternRecord]:
        """Get patterns by type across all agents."""
        query = select(BehavioralPatternRecord).where(
            BehavioralPatternRecord.pattern_type == pattern_type
        ).order_by(desc(BehavioralPatternRecord.statistical_significance))
        
        if limit:
            query = query.limit(limit)
        
        result = await session.execute(query)
        return list(result.scalars().all())