"""
Base repository class with common CRUD operations.
"""

from typing import TypeVar, Generic, List, Optional, Dict, Any
from abc import ABC, abstractmethod
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func
from sqlalchemy.orm import selectinload

T = TypeVar('T')


class BaseRepository(Generic[T], ABC):
    """Base repository with common CRUD operations."""
    
    def __init__(self, model_class: type[T]):
        self.model_class = model_class
    
    async def create(self, session: AsyncSession, **kwargs) -> T:
        """Create a new record."""
        instance = self.model_class(**kwargs)
        session.add(instance)
        await session.flush()
        await session.refresh(instance)
        return instance
    
    async def get_by_id(self, session: AsyncSession, id: UUID) -> Optional[T]:
        """Get a record by ID."""
        result = await session.execute(
            select(self.model_class).where(self.model_class.id == id)
        )
        return result.scalar_one_or_none()
    
    async def get_all(
        self,
        session: AsyncSession,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_by: Optional[str] = None
    ) -> List[T]:
        """Get all records with optional pagination and ordering."""
        query = select(self.model_class)
        
        if order_by:
            if hasattr(self.model_class, order_by):
                query = query.order_by(getattr(self.model_class, order_by))
        
        if offset:
            query = query.offset(offset)
        
        if limit:
            query = query.limit(limit)
        
        result = await session.execute(query)
        return result.scalars().all()
    
    async def update(
        self,
        session: AsyncSession,
        id: UUID,
        **kwargs
    ) -> Optional[T]:
        """Update a record by ID."""
        await session.execute(
            update(self.model_class)
            .where(self.model_class.id == id)
            .values(**kwargs)
        )
        return await self.get_by_id(session, id)
    
    async def delete(self, session: AsyncSession, id: UUID) -> bool:
        """Delete a record by ID."""
        result = await session.execute(
            delete(self.model_class).where(self.model_class.id == id)
        )
        return result.rowcount > 0
    
    async def count(self, session: AsyncSession, **filters) -> int:
        """Count records with optional filters."""
        query = select(func.count(self.model_class.id))
        
        for key, value in filters.items():
            if hasattr(self.model_class, key):
                query = query.where(getattr(self.model_class, key) == value)
        
        result = await session.execute(query)
        return result.scalar()
    
    async def exists(self, session: AsyncSession, **filters) -> bool:
        """Check if records exist with given filters."""
        query = select(self.model_class.id)
        
        for key, value in filters.items():
            if hasattr(self.model_class, key):
                query = query.where(getattr(self.model_class, key) == value)
        
        query = query.limit(1)
        result = await session.execute(query)
        return result.scalar_one_or_none() is not None
    
    async def find_by(
        self,
        session: AsyncSession,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_by: Optional[str] = None,
        **filters
    ) -> List[T]:
        """Find records by filters."""
        query = select(self.model_class)
        
        for key, value in filters.items():
            if hasattr(self.model_class, key):
                query = query.where(getattr(self.model_class, key) == value)
        
        if order_by:
            if hasattr(self.model_class, order_by):
                query = query.order_by(getattr(self.model_class, order_by))
        
        if offset:
            query = query.offset(offset)
        
        if limit:
            query = query.limit(limit)
        
        result = await session.execute(query)
        return result.scalars().all()
    
    async def find_one_by(self, session: AsyncSession, **filters) -> Optional[T]:
        """Find one record by filters."""
        results = await self.find_by(session, limit=1, **filters)
        return results[0] if results else None