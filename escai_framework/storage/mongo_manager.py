"""
MongoDB manager for coordinating repositories and operations.
"""

import logging
from typing import Dict, Any, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase

from .repositories.raw_log_repository import RawLogRepository
from .repositories.processed_event_repository import ProcessedEventRepository
from .repositories.explanation_repository import ExplanationRepository
from .repositories.configuration_repository import ConfigurationRepository
from .repositories.analytics_result_repository import AnalyticsResultRepository

logger = logging.getLogger(__name__)


class MongoManager:
    """Manager for MongoDB repositories and operations."""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        """Initialize MongoDB manager with database connection."""
        self.db = db
        
        # Initialize repositories
        self.raw_logs = RawLogRepository(db)
        self.processed_events = ProcessedEventRepository(db)
        self.explanations = ExplanationRepository(db)
        self.configurations = ConfigurationRepository(db)
        self.analytics_results = AnalyticsResultRepository(db)
        
        self._repositories = {
            'raw_logs': self.raw_logs,
            'processed_events': self.processed_events,
            'explanations': self.explanations,
            'configurations': self.configurations,
            'analytics_results': self.analytics_results
        }
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize all repositories and create indexes."""
        if self._initialized:
            logger.warning("MongoDB manager already initialized")
            return
        
        try:
            # Test database connection
            await self.db.command('ping')
            logger.info("MongoDB connection verified")
            
            # Create indexes for all repositories
            for name, repo in self._repositories.items():
                logger.info(f"Creating indexes for {name}")
                await repo.create_indexes()
            
            self._initialized = True
            logger.info("MongoDB manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB manager: {e}")
            raise
    
    async def get_database_info(self) -> Dict[str, Any]:
        """Get database information and statistics."""
        try:
            # Get database stats
            db_stats = await self.db.command('dbStats')
            
            # Get collection stats
            collection_stats = {}
            for name, repo in self._repositories.items():
                try:
                    stats = await self.db.command('collStats', repo.collection_name)
                    collection_stats[name] = {
                        'count': stats.get('count', 0),
                        'size': stats.get('size', 0),
                        'avgObjSize': stats.get('avgObjSize', 0),
                        'storageSize': stats.get('storageSize', 0),
                        'indexes': stats.get('nindexes', 0),
                        'indexSize': stats.get('totalIndexSize', 0)
                    }
                except Exception as e:
                    logger.warning(f"Could not get stats for {name}: {e}")
                    collection_stats[name] = {'error': str(e)}
            
            return {
                'database': {
                    'name': self.db.name,
                    'collections': db_stats.get('collections', 0),
                    'dataSize': db_stats.get('dataSize', 0),
                    'storageSize': db_stats.get('storageSize', 0),
                    'indexes': db_stats.get('indexes', 0),
                    'indexSize': db_stats.get('indexSize', 0)
                },
                'collections': collection_stats
            }
            
        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return {'error': str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on MongoDB and all repositories."""
        health_status = {
            'mongodb': {'status': 'unknown'},
            'repositories': {}
        }
        
        try:
            # Test MongoDB connection
            await self.db.command('ping')
            health_status['mongodb'] = {'status': 'healthy'}
            
            # Test each repository
            for name, repo in self._repositories.items():
                try:
                    # Try to count documents (lightweight operation)
                    count = await repo.count_documents({})
                    health_status['repositories'][name] = {
                        'status': 'healthy',
                        'document_count': count
                    }
                except Exception as e:
                    health_status['repositories'][name] = {
                        'status': 'unhealthy',
                        'error': str(e)
                    }
            
        except Exception as e:
            health_status['mongodb'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
        
        return health_status
    
    async def cleanup_old_data(
        self,
        raw_logs_days: int = 30,
        processed_events_days: int = 90,
        explanations_days: int = 180,
        analytics_results_days: int = 365
    ) -> Dict[str, int]:
        """Cleanup old data from all collections."""
        cleanup_results = {}
        
        try:
            # Cleanup raw logs
            deleted = await self.raw_logs.cleanup_old_logs(raw_logs_days)
            cleanup_results['raw_logs'] = deleted
            logger.info(f"Cleaned up {deleted} old raw logs")
            
            # Cleanup processed events
            deleted = await self.processed_events.cleanup_old_events(processed_events_days)
            cleanup_results['processed_events'] = deleted
            logger.info(f"Cleaned up {deleted} old processed events")
            
            # Cleanup explanations
            deleted = await self.explanations.cleanup_old_explanations(explanations_days)
            cleanup_results['explanations'] = deleted
            logger.info(f"Cleaned up {deleted} old explanations")
            
            # Cleanup analytics results
            deleted = await self.analytics_results.cleanup_old_results(analytics_results_days)
            cleanup_results['analytics_results'] = deleted
            logger.info(f"Cleaned up {deleted} old analytics results")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            cleanup_results['error'] = str(e)
        
        return cleanup_results
    
    async def optimize_collections(self) -> Dict[str, Any]:
        """Optimize all collections (rebuild indexes, compact, etc.)."""
        optimization_results = {}
        
        for name, repo in self._repositories.items():
            try:
                # Recreate indexes
                await repo.create_indexes()
                
                # Get collection stats after optimization
                stats = await self.db.command('collStats', repo.collection_name)
                optimization_results[name] = {
                    'status': 'optimized',
                    'document_count': stats.get('count', 0),
                    'index_count': stats.get('nindexes', 0)
                }
                
            except Exception as e:
                optimization_results[name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return optimization_results
    
    async def backup_collection_schemas(self) -> Dict[str, Any]:
        """Backup collection schemas and index definitions."""
        schemas = {}
        
        for name, repo in self._repositories.items():
            try:
                # Get collection info
                collection_info = await self.db.command('listCollections', filter={'name': repo.collection_name})
                
                # Get indexes
                indexes = []
                async for index in repo.collection.list_indexes():
                    indexes.append(index)
                
                schemas[name] = {
                    'collection_info': collection_info,
                    'indexes': indexes
                }
                
            except Exception as e:
                schemas[name] = {'error': str(e)}
        
        return schemas
    
    async def get_repository(self, name: str):
        """Get a specific repository by name."""
        if name not in self._repositories:
            raise ValueError(f"Repository '{name}' not found")
        return self._repositories[name]
    
    @property
    def is_initialized(self) -> bool:
        """Check if the manager is initialized."""
        return self._initialized
    
    @property
    def repository_names(self) -> list:
        """Get list of available repository names."""
        return list(self._repositories.keys())