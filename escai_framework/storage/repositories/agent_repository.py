"""
Repository for Agent model operations.
"""

from typing import List, Optional
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from .base_repository import BaseRepository
from ..models import Agent


class AgentRepository(BaseRepository[Agent]):
    """Repository for Agent operations."""
    
    def __init__(self):
        super().__init__(Agent)
    
    async def get_by_agent_id(self, session: AsyncSession, agent_id: str) -> Optional[Agent]:
        """Get agent by agent_id."""
        result = await session.execute(
            select(Agent).where(Agent.agent_id == agent_id)
        )
        return result.scalar_one_or_none()
    
    async def get_active_agents(self, session: AsyncSession) -> List[Agent]:
        """Get all active agents."""
        return await self.find_by(session, is_active=True, order_by='created_at')
    
    async def get_by_framework(
        self,
        session: AsyncSession,
        framework: str,
        active_only: bool = True
    ) -> List[Agent]:
        """Get agents by framework."""
        filters = {'framework': framework}
        if active_only:
            filters['is_active'] = True
        
        return await self.find_by(session, order_by='created_at', **filters)
    
    async def deactivate_agent(self, session: AsyncSession, agent_id: str) -> bool:
        """Deactivate an agent."""
        agent = await self.get_by_agent_id(session, agent_id)
        if agent:
            agent.is_active = False
            agent.updated_at = datetime.utcnow()
            await session.flush()
            return True
        return False
    
    async def activate_agent(self, session: AsyncSession, agent_id: str) -> bool:
        """Activate an agent."""
        agent = await self.get_by_agent_id(session, agent_id)
        if agent:
            agent.is_active = True
            agent.updated_at = datetime.utcnow()
            await session.flush()
            return True
        return False
    
    async def update_configuration(
        self,
        session: AsyncSession,
        agent_id: str,
        configuration: dict
    ) -> Optional[Agent]:
        """Update agent configuration."""
        agent = await self.get_by_agent_id(session, agent_id)
        if agent:
            agent.configuration = configuration
            agent.updated_at = datetime.utcnow()
            await session.flush()
            await session.refresh(agent)
            return agent
        return None