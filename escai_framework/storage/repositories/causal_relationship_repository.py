"""
Repository for CausalRelationshipRecord model operations.
"""

from typing import List, Optional
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc, asc, func, or_

from .base_repository import BaseRepository
from ..models import CausalRelationshipRecord


class CausalRelationshipRepository(BaseRepository[CausalRelationshipRecord]):
    """Repository for CausalRelationshipRecord operations."""
    
    def __init__(self):
        super().__init__(CausalRelationshipRecord)
    
    async def get_by_relationship_id(
        self,
        session: AsyncSession,
        relationship_id: str
    ) -> Optional[CausalRelationshipRecord]:
        """Get relationship by relationship_id."""
        result = await session.execute(
            select(CausalRelationshipRecord)
            .where(CausalRelationshipRecord.relationship_id == relationship_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_cause_event(
        self,
        session: AsyncSession,
        cause_event: str,
        min_strength: float = 0.0,
        limit: Optional[int] = None
    ) -> List[CausalRelationshipRecord]:
        """Get relationships by cause event."""
        query = select(CausalRelationshipRecord).where(
            and_(
                CausalRelationshipRecord.cause_event == cause_event,
                CausalRelationshipRecord.strength >= min_strength
            )
        ).order_by(desc(CausalRelationshipRecord.strength))
        
        if limit:
            query = query.limit(limit)
        
        result = await session.execute(query)
        return result.scalars().all()
    
    async def get_by_effect_event(
        self,
        session: AsyncSession,
        effect_event: str,
        min_strength: float = 0.0,
        limit: Optional[int] = None
    ) -> List[CausalRelationshipRecord]:
        """Get relationships by effect event."""
        query = select(CausalRelationshipRecord).where(
            and_(
                CausalRelationshipRecord.effect_event == effect_event,
                CausalRelationshipRecord.strength >= min_strength
            )
        ).order_by(desc(CausalRelationshipRecord.strength))
        
        if limit:
            query = query.limit(limit)
        
        result = await session.execute(query)
        return result.scalars().all()
    
    async def get_strong_relationships(
        self,
        session: AsyncSession,
        min_strength: float = 0.7,
        min_confidence: float = 0.8,
        limit: Optional[int] = None
    ) -> List[CausalRelationshipRecord]:
        """Get strong causal relationships."""
        query = select(CausalRelationshipRecord).where(
            and_(
                CausalRelationshipRecord.strength >= min_strength,
                CausalRelationshipRecord.confidence >= min_confidence
            )
        ).order_by(desc(CausalRelationshipRecord.strength))
        
        if limit:
            query = query.limit(limit)
        
        result = await session.execute(query)
        return result.scalars().all()
    
    async def get_by_analysis_method(
        self,
        session: AsyncSession,
        analysis_method: str,
        limit: Optional[int] = None
    ) -> List[CausalRelationshipRecord]:
        """Get relationships by analysis method."""
        query = select(CausalRelationshipRecord).where(
            CausalRelationshipRecord.analysis_method == analysis_method
        ).order_by(desc(CausalRelationshipRecord.discovered_at))
        
        if limit:
            query = query.limit(limit)
        
        result = await session.execute(query)
        return result.scalars().all()
    
    async def get_recent_discoveries(
        self,
        session: AsyncSession,
        days: int = 7,
        limit: Optional[int] = None
    ) -> List[CausalRelationshipRecord]:
        """Get recently discovered relationships."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        query = select(CausalRelationshipRecord).where(
            CausalRelationshipRecord.discovered_at >= cutoff_date
        ).order_by(desc(CausalRelationshipRecord.discovered_at))
        
        if limit:
            query = query.limit(limit)
        
        result = await session.execute(query)
        return result.scalars().all()
    
    async def search_relationships(
        self,
        session: AsyncSession,
        search_term: str,
        limit: Optional[int] = None
    ) -> List[CausalRelationshipRecord]:
        """Search relationships by cause or effect event."""
        query = select(CausalRelationshipRecord).where(
            or_(
                CausalRelationshipRecord.cause_event.ilike(f'%{search_term}%'),
                CausalRelationshipRecord.effect_event.ilike(f'%{search_term}%')
            )
        ).order_by(desc(CausalRelationshipRecord.strength))
        
        if limit:
            query = query.limit(limit)
        
        result = await session.execute(query)
        return result.scalars().all()
    
    async def get_causal_chain(
        self,
        session: AsyncSession,
        start_event: str,
        max_depth: int = 3
    ) -> List[CausalRelationshipRecord]:
        """Get causal chain starting from an event."""
        relationships = []
        current_events = [start_event]
        
        for depth in range(max_depth):
            if not current_events:
                break
            
            # Get relationships where current events are causes
            next_relationships = []
            next_events = []
            
            for event in current_events:
                event_relationships = await self.get_by_cause_event(session, event)
                next_relationships.extend(event_relationships)
                next_events.extend([r.effect_event for r in event_relationships])
            
            if not next_relationships:
                break
            
            relationships.extend(next_relationships)
            current_events = list(set(next_events))  # Remove duplicates
        
        return relationships
    
    async def get_relationship_statistics(self, session: AsyncSession) -> dict:
        """Get causal relationship statistics."""
        result = await session.execute(
            select(
                func.count(CausalRelationshipRecord.id).label('total_relationships'),
                func.avg(CausalRelationshipRecord.strength).label('avg_strength'),
                func.avg(CausalRelationshipRecord.confidence).label('avg_confidence'),
                func.avg(CausalRelationshipRecord.delay_ms).label('avg_delay_ms'),
                func.count(
                    CausalRelationshipRecord.id
                ).filter(
                    CausalRelationshipRecord.strength >= 0.7
                ).label('strong_relationships'),
                func.count(
                    CausalRelationshipRecord.id
                ).filter(
                    CausalRelationshipRecord.confidence >= 0.8
                ).label('high_confidence_relationships')
            )
        )
        row = result.first()
        return {
            'total_relationships': row.total_relationships,
            'avg_strength': float(row.avg_strength) if row.avg_strength else 0.0,
            'avg_confidence': float(row.avg_confidence) if row.avg_confidence else 0.0,
            'avg_delay_ms': float(row.avg_delay_ms) if row.avg_delay_ms else 0.0,
            'strong_relationships': row.strong_relationships,
            'high_confidence_relationships': row.high_confidence_relationships
        }