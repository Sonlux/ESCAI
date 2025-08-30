"""
Repository for MonitoringSession model operations.
"""

from typing import List, Optional
from datetime import datetime, timedelta
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc, func

from .base_repository import BaseRepository
from ..models import MonitoringSession


class MonitoringSessionRepository(BaseRepository[MonitoringSession]):
    """Repository for MonitoringSession operations."""

    def __init__(self):
        super().__init__(MonitoringSession)

    async def get_by_session_id(
        self,
        session: AsyncSession,
        session_id: str
    ) -> Optional[MonitoringSession]:
        """Get session by session_id."""
        result = await session.execute(
            select(MonitoringSession)
            .where(MonitoringSession.session_id == session_id)
        )
        return result.scalar_one_or_none()

    async def get_active_sessions(
        self,
        session: AsyncSession,
        agent_id: Optional[UUID] = None
    ) -> List[MonitoringSession]:
        """Get active monitoring sessions."""
        query = select(MonitoringSession).where(
            MonitoringSession.status == 'active'
        )

        if agent_id:
            query = query.where(MonitoringSession.agent_id == agent_id)

        query = query.order_by(desc(MonitoringSession.started_at))

        result = await session.execute(query)
        return list(result.scalars().all())

    async def get_by_agent(
        self,
        session: AsyncSession,
        agent_id: UUID,
        status: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[MonitoringSession]:
        """Get sessions for an agent."""
        query = select(MonitoringSession).where(
            MonitoringSession.agent_id == agent_id
        )

        if status:
            query = query.where(MonitoringSession.status == status)

        query = query.order_by(desc(MonitoringSession.started_at))

        if limit:
            query = query.limit(limit)

        result = await session.execute(query)
        return list(result.scalars().all())

    async def get_recent_sessions(
        self,
        session: AsyncSession,
        hours: int = 24,
        status: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[MonitoringSession]:
        """Get recent monitoring sessions."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        query = select(MonitoringSession).where(
            MonitoringSession.started_at >= cutoff_time
        )

        if status:
            query = query.where(MonitoringSession.status == status)

        query = query.order_by(desc(MonitoringSession.started_at))

        if limit:
            query = query.limit(limit)

        result = await session.execute(query)
        return list(result.scalars().all())

    async def end_session(
        self,
        session: AsyncSession,
        session_id: str,
        status: str = 'completed',
        metadata: Optional[dict] = None
    ) -> Optional[MonitoringSession]:
        """End a monitoring session."""
        monitoring_session = await self.get_by_session_id(session, session_id)
        if monitoring_session and monitoring_session.status == 'active':
            monitoring_session.ended_at = datetime.utcnow()  # type: ignore
            monitoring_session.status = status  # type: ignore

            if metadata:
                if monitoring_session.session_metadata:
                    monitoring_session.session_metadata.update(
                        metadata)  # type: ignore
                else:
                    monitoring_session.session_metadata = (
                        metadata)  # type: ignore

            await session.flush()
            await session.refresh(monitoring_session)
            return monitoring_session
        return None

    async def get_session_duration_stats(
        self,
        session: AsyncSession,
        agent_id: UUID,
        days: int = 30
    ) -> dict:
        """Get session duration statistics for an agent."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Get completed sessions with duration
        result = await session.execute(
            select(
                func.count(MonitoringSession.id).label('total_sessions'),
                func.avg(
                    func.extract(
                        'epoch',
                        (MonitoringSession.ended_at -
                         MonitoringSession.started_at)
                    )
                ).label('avg_duration_seconds'),
                func.min(
                    func.extract(
                        'epoch',
                        (MonitoringSession.ended_at -
                         MonitoringSession.started_at)
                    )
                ).label('min_duration_seconds'),
                func.max(
                    func.extract(
                        'epoch',
                        (MonitoringSession.ended_at -
                         MonitoringSession.started_at)
                    )
                ).label('max_duration_seconds')
            )
            .where(
                and_(
                    MonitoringSession.agent_id == agent_id,
                    MonitoringSession.started_at >= cutoff_date,
                    MonitoringSession.ended_at.isnot(None),
                    MonitoringSession.status == 'completed'
                )
            )
        )
        row = result.first()

        return {
            'total_sessions': row.total_sessions,
            'avg_duration_seconds': (
                float(row.avg_duration_seconds)
                if row.avg_duration_seconds else 0.0
            ),
            'min_duration_seconds': (
                float(row.min_duration_seconds)
                if row.min_duration_seconds else 0.0
            ),
            'max_duration_seconds': (
                float(row.max_duration_seconds)
                if row.max_duration_seconds else 0.0
            )
        }

    async def get_session_status_counts(
        self,
        session: AsyncSession,
        days: int = 30
    ) -> dict:
        """Get session status counts."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        result = await session.execute(
            select(
                MonitoringSession.status,
                func.count(MonitoringSession.id).label('count')
            )
            .where(MonitoringSession.started_at >= cutoff_date)
            .group_by(MonitoringSession.status)
        )

        status_counts = {row.status: int(row.count) for row in result}

        # Ensure all statuses are represented
        for status in ['active', 'completed', 'failed']:
            if status not in status_counts:
                status_counts[status] = 0

        return status_counts

    async def cleanup_old_sessions(
        self,
        session: AsyncSession,
        days_to_keep: int = 90
    ) -> int:
        """Clean up old completed sessions."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

        result = await session.execute(
            select(func.count(MonitoringSession.id))
            .where(
                and_(
                    MonitoringSession.started_at < cutoff_date,
                    MonitoringSession.status.in_(['completed', 'failed'])
                )
            )
        )
        count_to_delete = result.scalar()

        await session.execute(
            MonitoringSession.__table__.delete()
            .where(
                and_(
                    MonitoringSession.started_at < cutoff_date,
                    MonitoringSession.status.in_(['completed', 'failed'])
                )
            )
        )

        return count_to_delete
