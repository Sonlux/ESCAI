"""
Repository for system and user configurations in MongoDB.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from motor.motor_asyncio import AsyncIOMotorDatabase

from .mongo_base_repository import MongoBaseRepository
from ..mongo_models import ConfigurationDocument

logger = logging.getLogger(__name__)


class ConfigurationRepository(MongoBaseRepository[ConfigurationDocument]):
    """Repository for managing system and user configurations."""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db, "configurations", ConfigurationDocument)
    
    async def create_indexes(self):
        """Create indexes optimized for configuration queries."""
        await super().create_indexes()
        
        # Unique compound index for configuration identification
        await self.collection.create_index([
            ("config_type", 1),
            ("config_name", 1),
            ("user_id", 1),
            ("agent_id", 1)
        ], unique=True, sparse=True)
        
        # Index for active configurations
        await self.collection.create_index([
            ("is_active", 1),
            ("config_type", 1)
        ])
        
        # Index for user-specific configurations
        await self.collection.create_index([
            ("user_id", 1),
            ("is_active", 1)
        ])
        
        # Index for agent-specific configurations
        await self.collection.create_index([
            ("agent_id", 1),
            ("is_active", 1)
        ])
        
        # Index for version management
        await self.collection.create_index([
            ("config_type", 1),
            ("config_name", 1),
            ("version", -1)
        ])
        
        logger.info("Created indexes for configurations collection")
    
    async def find_by_type(
        self,
        config_type: str,
        is_active: bool = True
    ) -> List[ConfigurationDocument]:
        """Find configurations by type."""
        filter_dict = {
            "config_type": config_type,
            "is_active": is_active
        }
        
        return await self.find_many(
            filter_dict,
            sort=[("config_name", 1), ("version", -1)]
        )
    
    async def find_by_name(
        self,
        config_type: str,
        config_name: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        is_active: bool = True
    ) -> Optional[ConfigurationDocument]:
        """Find a specific configuration by name."""
        filter_dict = {
            "config_type": config_type,
            "config_name": config_name,
            "is_active": is_active
        }
        
        if user_id:
            filter_dict["user_id"] = user_id
        if agent_id:
            filter_dict["agent_id"] = agent_id
        
        # Find the latest version
        configs = await self.find_many(
            filter_dict,
            sort=[("version", -1)],
            limit=1
        )
        
        return configs[0] if configs else None
    
    async def find_user_configurations(
        self,
        user_id: str,
        config_type: Optional[str] = None,
        is_active: bool = True
    ) -> List[ConfigurationDocument]:
        """Find configurations for a specific user."""
        filter_dict = {
            "user_id": user_id,
            "is_active": is_active
        }
        
        if config_type:
            filter_dict["config_type"] = config_type
        
        return await self.find_many(
            filter_dict,
            sort=[("config_type", 1), ("config_name", 1)]
        )
    
    async def find_agent_configurations(
        self,
        agent_id: str,
        config_type: Optional[str] = None,
        is_active: bool = True
    ) -> List[ConfigurationDocument]:
        """Find configurations for a specific agent."""
        filter_dict = {
            "agent_id": agent_id,
            "is_active": is_active
        }
        
        if config_type:
            filter_dict["config_type"] = config_type
        
        return await self.find_many(
            filter_dict,
            sort=[("config_type", 1), ("config_name", 1)]
        )
    
    async def find_system_configurations(
        self,
        config_type: Optional[str] = None,
        is_active: bool = True
    ) -> List[ConfigurationDocument]:
        """Find system-wide configurations."""
        filter_dict = {
            "config_type": config_type or "system",
            "user_id": None,
            "agent_id": None,
            "is_active": is_active
        }
        
        return await self.find_many(
            filter_dict,
            sort=[("config_name", 1)]
        )
    
    async def create_or_update_configuration(
        self,
        config_type: str,
        config_name: str,
        config_data: Dict[str, Any],
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None
    ) -> str:
        """Create a new configuration or update existing one."""
        # Find existing configuration
        existing = await self.find_by_name(
            config_type, config_name, user_id, agent_id, is_active=True
        )
        
        if existing:
            # Create new version
            new_version = existing.version + 1
            
            # Deactivate old version
            await self.update_by_id(
                str(existing.id),
                {"$set": {"is_active": False}}
            )
        else:
            new_version = 1
        
        # Create new configuration
        new_config = ConfigurationDocument(
            config_type=config_type,
            config_name=config_name,
            config_data=config_data,
            user_id=user_id,
            agent_id=agent_id,
            version=new_version,
            is_active=True
        )
        
        return await self.insert_one(new_config)
    
    async def deactivate_configuration(
        self,
        config_type: str,
        config_name: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None
    ) -> bool:
        """Deactivate a configuration."""
        filter_dict = {
            "config_type": config_type,
            "config_name": config_name,
            "is_active": True
        }
        
        if user_id:
            filter_dict["user_id"] = user_id
        if agent_id:
            filter_dict["agent_id"] = agent_id
        
        return await self.update_one(
            filter_dict,
            {"$set": {"is_active": False}}
        )
    
    async def get_configuration_history(
        self,
        config_type: str,
        config_name: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 10
    ) -> List[ConfigurationDocument]:
        """Get configuration version history."""
        filter_dict = {
            "config_type": config_type,
            "config_name": config_name
        }
        
        if user_id:
            filter_dict["user_id"] = user_id
        if agent_id:
            filter_dict["agent_id"] = agent_id
        
        return await self.find_many(
            filter_dict,
            sort=[("version", -1)],
            limit=limit
        )
    
    async def restore_configuration_version(
        self,
        config_type: str,
        config_name: str,
        version: int,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None
    ) -> Optional[str]:
        """Restore a specific configuration version."""
        # Find the version to restore
        filter_dict = {
            "config_type": config_type,
            "config_name": config_name,
            "version": version
        }
        
        if user_id:
            filter_dict["user_id"] = user_id
        if agent_id:
            filter_dict["agent_id"] = agent_id
        
        version_to_restore = await self.find_one(filter_dict)
        if not version_to_restore:
            return None
        
        # Create new configuration with restored data
        return await self.create_or_update_configuration(
            config_type,
            config_name,
            version_to_restore.config_data,
            user_id,
            agent_id
        )
    
    async def get_configuration_statistics(self) -> Dict[str, Any]:
        """Get configuration statistics."""
        pipeline: List[Dict[str, Any]] = [
            {
                "$group": {
                    "_id": {
                        "config_type": "$config_type",
                        "is_active": "$is_active"
                    },
                    "count": {"$sum": 1},
                    "latest_updated": {"$max": "$updated_at"}
                }
            },
            {
                "$group": {
                    "_id": "$_id.config_type",
                    "active_count": {
                        "$sum": {
                            "$cond": [{"$eq": ["$_id.is_active", True]}, "$count", 0]
                        }
                    },
                    "inactive_count": {
                        "$sum": {
                            "$cond": [{"$eq": ["$_id.is_active", False]}, "$count", 0]
                        }
                    },
                    "total_count": {"$sum": "$count"},
                    "latest_updated": {"$max": "$latest_updated"}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "by_type": {
                        "$push": {
                            "config_type": "$_id",
                            "active_count": "$active_count",
                            "inactive_count": "$inactive_count",
                            "total_count": "$total_count",
                            "latest_updated": "$latest_updated"
                        }
                    },
                    "total_configurations": {"$sum": "$total_count"},
                    "total_active": {"$sum": "$active_count"}
                }
            }
        ]
        
        result = await self.aggregate(pipeline)
        if result:
            return result[0]
        return {"by_type": [], "total_configurations": 0, "total_active": 0}
    
    async def cleanup_old_versions(
        self,
        keep_versions: int = 5,
        config_type: Optional[str] = None
    ) -> int:
        """Cleanup old configuration versions, keeping only the latest N versions."""
        match_stage: Dict[str, Any] = {"is_active": False}
        if config_type:
            match_stage["config_type"] = config_type
        
        pipeline: List[Dict[str, Any]] = [
            {"$match": match_stage},
            {
                "$group": {
                    "_id": {
                        "config_type": "$config_type",
                        "config_name": "$config_name",
                        "user_id": "$user_id",
                        "agent_id": "$agent_id"
                    },
                    "versions": {
                        "$push": {
                            "id": "$_id",
                            "version": "$version"
                        }
                    }
                }
            },
            {
                "$project": {
                    "versions_to_delete": {
                        "$slice": [
                            {
                                "$sortArray": {
                                    "input": "$versions",
                                    "sortBy": {"version": -1}
                                }
                            },
                            keep_versions,
                            {"$size": "$versions"}
                        ]
                    }
                }
            }
        ]
        
        result = await self.aggregate(pipeline)
        
        deleted_count = 0
        for doc in result:
            for version_info in doc.get("versions_to_delete", []):
                if await self.delete_by_id(str(version_info["id"])):
                    deleted_count += 1
        
        return deleted_count