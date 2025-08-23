"""
Database connection management and configuration for ESCAI Framework.
"""

import os
from typing import Optional, AsyncGenerator, Dict, Any
from contextlib import asynccontextmanager
import logging

from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from .base import Base
from .mongo_manager import MongoManager
from .redis_manager import RedisManager

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database connections and sessions with connection pooling."""
    
    def __init__(self):
        self._async_engine = None
        self._sync_engine = None
        self._async_session_factory = None
        self._sync_session_factory = None
        self._mongo_client = None
        self._async_mongo_client = None
        self._mongo_db = None
        self._async_mongo_db = None
        self._mongo_manager = None
        self._redis_manager = None
        self._initialized = False
    
    def initialize(
        self,
        database_url: Optional[str] = None,
        async_database_url: Optional[str] = None,
        mongo_url: Optional[str] = None,
        mongo_db_name: Optional[str] = None,
        redis_url: Optional[str] = None,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_timeout: int = 30,
        pool_recycle: int = 3600
    ):
        """Initialize database connections with connection pooling."""
        if self._initialized:
            logger.warning("Database manager already initialized")
            return
        
        # Use environment variables if URLs not provided
        if not database_url:
            database_url = os.getenv(
                'ESCAI_DATABASE_URL',
                'postgresql://escai:escai@localhost:5432/escai'
            )
        
        if not async_database_url:
            async_database_url = os.getenv(
                'ESCAI_ASYNC_DATABASE_URL',
                'postgresql+asyncpg://escai:escai@localhost:5432/escai'
            )
        
        if not mongo_url:
            mongo_url = os.getenv(
                'ESCAI_MONGO_URL',
                'mongodb://localhost:27017'
            )
        
        if not mongo_db_name:
            mongo_db_name = os.getenv(
                'ESCAI_MONGO_DB_NAME',
                'escai_unstructured'
            )
        
        # Create synchronous engine
        sync_engine_kwargs = {
            'echo': os.getenv('ESCAI_DB_ECHO', 'false').lower() == 'true'
        }
        
        # Add pooling parameters only for non-SQLite databases
        if not database_url.startswith('sqlite'):
            sync_engine_kwargs.update({
                'poolclass': QueuePool,
                'pool_size': pool_size,
                'max_overflow': max_overflow,
                'pool_timeout': pool_timeout,
                'pool_recycle': pool_recycle
            })
        
        self._sync_engine = create_engine(database_url, **sync_engine_kwargs)
        
        # Create asynchronous engine
        async_engine_kwargs = {
            'echo': os.getenv('ESCAI_DB_ECHO', 'false').lower() == 'true'
        }
        
        # Add pooling parameters only for non-SQLite databases
        if not async_database_url.startswith('sqlite'):
            async_engine_kwargs.update({
                'pool_size': pool_size,
                'max_overflow': max_overflow,
                'pool_timeout': pool_timeout,
                'pool_recycle': pool_recycle
            })
        
        self._async_engine = create_async_engine(async_database_url, **async_engine_kwargs)
        
        # Create session factories
        self._sync_session_factory = sessionmaker(
            bind=self._sync_engine,
            expire_on_commit=False
        )
        
        self._async_session_factory = async_sessionmaker(
            bind=self._async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Initialize MongoDB connections
        self._init_mongodb(mongo_url, mongo_db_name, pool_size)
        
        # Initialize Redis connections
        self._init_redis(redis_url, pool_size)
        
        self._initialized = True
        logger.info("Database manager initialized successfully")
    
    def _init_mongodb(self, mongo_url: str, mongo_db_name: str, pool_size: int):
        """Initialize MongoDB connections."""
        try:
            # Synchronous MongoDB client
            self._mongo_client = MongoClient(
                mongo_url,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000,
                maxPoolSize=pool_size,
                minPoolSize=1
            )
            
            # Test connection
            self._mongo_client.admin.command('ping')
            self._mongo_db = self._mongo_client[mongo_db_name]
            
            # Asynchronous MongoDB client
            self._async_mongo_client = AsyncIOMotorClient(
                mongo_url,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000,
                maxPoolSize=pool_size,
                minPoolSize=1
            )
            
            self._async_mongo_db = self._async_mongo_client[mongo_db_name]
            
            # Initialize MongoDB manager
            self._mongo_manager = MongoManager(self._async_mongo_db)
            
            logger.info("MongoDB connections initialized successfully")
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            # Continue without MongoDB for now - can be made optional
            self._mongo_client = None
            self._async_mongo_client = None
            self._mongo_db = None
            self._async_mongo_db = None
    
    def _init_redis(self, redis_url: Optional[str], pool_size: int):
        """Initialize Redis connections."""
        try:
            self._redis_manager = RedisManager()
            self._redis_manager.initialize(
                redis_url=redis_url,
                max_connections=pool_size,
                failover_enabled=True
            )
            logger.info("Redis connections initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            # Continue without Redis for graceful degradation
            self._redis_manager = None
    
    async def test_mongo_connection(self) -> bool:
        """Test MongoDB connection asynchronously."""
        if not self._async_mongo_client:
            return False
        try:
            await self._async_mongo_client.admin.command('ping')
            return True
        except Exception as e:
            logger.error(f"MongoDB connection test failed: {e}")
            return False
    
    async def test_redis_connection(self) -> bool:
        """Test Redis connection asynchronously."""
        if not self._redis_manager:
            return False
        return await self._redis_manager.test_connection()
    
    @property
    def async_engine(self):
        """Get the async database engine."""
        if not self._initialized:
            raise RuntimeError("Database manager not initialized")
        return self._async_engine
    
    @property
    def sync_engine(self):
        """Get the sync database engine."""
        if not self._initialized:
            raise RuntimeError("Database manager not initialized")
        return self._sync_engine
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an async database session with automatic cleanup."""
        if not self._initialized:
            raise RuntimeError("Database manager not initialized")
        
        async with self._async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    def get_sync_session(self):
        """Get a sync database session."""
        if not self._initialized:
            raise RuntimeError("Database manager not initialized")
        return self._sync_session_factory()
    
    @property
    def mongo_db(self):
        """Get the synchronous MongoDB database."""
        if not self._initialized:
            raise RuntimeError("Database manager not initialized")
        if not self._mongo_db:
            raise RuntimeError("MongoDB not available")
        return self._mongo_db
    
    @property
    def async_mongo_db(self):
        """Get the asynchronous MongoDB database."""
        if not self._initialized:
            raise RuntimeError("Database manager not initialized")
        if not self._async_mongo_db:
            raise RuntimeError("Async MongoDB not available")
        return self._async_mongo_db
    
    @property
    def mongo_available(self) -> bool:
        """Check if MongoDB is available."""
        return self._mongo_db is not None and self._async_mongo_db is not None
    
    @property
    def mongo_manager(self) -> MongoManager:
        """Get the MongoDB manager."""
        if not self._initialized:
            raise RuntimeError("Database manager not initialized")
        if not self._mongo_manager:
            raise RuntimeError("MongoDB manager not available")
        return self._mongo_manager
    
    @property
    def redis_available(self) -> bool:
        """Check if Redis is available."""
        return self._redis_manager is not None and self._redis_manager.available
    
    @property
    def redis_manager(self) -> RedisManager:
        """Get the Redis manager."""
        if not self._initialized:
            raise RuntimeError("Database manager not initialized")
        if not self._redis_manager:
            raise RuntimeError("Redis manager not available")
        return self._redis_manager
    
    async def create_tables(self):
        """Create all database tables."""
        if not self._initialized:
            raise RuntimeError("Database manager not initialized")
        
        async with self._async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")
        
        # Initialize MongoDB manager if available
        if self._mongo_manager:
            await self._mongo_manager.initialize()
            logger.info("MongoDB collections and indexes created successfully")
    
    async def drop_tables(self):
        """Drop all database tables."""
        if not self._initialized:
            raise RuntimeError("Database manager not initialized")
        
        async with self._async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.info("Database tables dropped successfully")
    
    async def close(self):
        """Close database connections."""
        if self._async_engine:
            await self._async_engine.dispose()
        if self._sync_engine:
            self._sync_engine.dispose()
        if self._mongo_client:
            self._mongo_client.close()
        if self._async_mongo_client:
            self._async_mongo_client.close()
        if self._redis_manager:
            await self._redis_manager.close()
        self._initialized = False
        logger.info("Database connections closed")

# Global database manager instance
db_manager = DatabaseManager()