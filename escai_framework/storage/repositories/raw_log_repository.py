"""
Repository for raw agent execution logs in MongoDB.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
from motor.motor_asyncio import AsyncIOMotorDatabase

from .mongo_base_repository import MongoBaseRepository
from ..mongo_models import RawLogDocument

logger = logging.getLogger(__name__)


class RawLogRepository(MongoBaseRepository[RawLogDocument]):
    """Repository for managing raw agent execution logs."""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db, "raw_logs", RawLogDocument)
    
    async def create_indexes(self):
        """Create indexes optimized for log queries."""
        await super().create_indexes()
        
        # Compound indexes for common queries
        await self.collection.create_index([
            ("agent_id", 1),
            ("timestamp", -1)
        ])
        
        await self.collection.create_index([
            ("session_id", 1),
            ("timestamp", -1)
        ])
        
        await self.collection.create_index([
            ("framework", 1),
            ("timestamp", -1)
        ])
        
        await self.collection.create_index([
            ("log_level", 1),
            ("timestamp", -1)
        ])
        
        # Text index for log message search
        await self.create_text_index(["message"])
        
        # TTL index for automatic log cleanup (30 days)
        await self.collection.create_index(
            "timestamp",
            expireAfterSeconds=30 * 24 * 60 * 60  # 30 days
        )
        
        logger.info("Created indexes for raw_logs collection")
    
    async def find_by_agent(
        self,
        agent_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        log_level: Optional[str] = None,
        limit: int = 1000
    ) -> List[RawLogDocument]:
        """Find logs for a specific agent."""
        filter_dict: Dict[str, Any] = {"agent_id": agent_id}
        
        if start_time or end_time:
            time_filter = {}
            if start_time:
                time_filter["$gte"] = start_time
            if end_time:
                time_filter["$lte"] = end_time
            filter_dict["timestamp"] = time_filter
        
        if log_level:
            filter_dict["log_level"] = log_level.upper()
        
        return await self.find_many(
            filter_dict,
            sort=[("timestamp", -1)],
            limit=limit
        )
    
    async def find_by_session(
        self,
        session_id: str,
        log_level: Optional[str] = None,
        limit: int = 1000
    ) -> List[RawLogDocument]:
        """Find logs for a specific monitoring session."""
        filter_dict: Dict[str, Any] = {"session_id": session_id}
        
        if log_level:
            filter_dict["log_level"] = log_level.upper()
        
        return await self.find_many(
            filter_dict,
            sort=[("timestamp", -1)],
            limit=limit
        )
    
    async def find_by_framework(
        self,
        framework: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[RawLogDocument]:
        """Find logs for a specific framework."""
        filter_dict: Dict[str, Any] = {"framework": framework.lower()}
        
        if start_time or end_time:
            time_filter = {}
            if start_time:
                time_filter["$gte"] = start_time
            if end_time:
                time_filter["$lte"] = end_time
            filter_dict["timestamp"] = time_filter
        
        return await self.find_many(
            filter_dict,
            sort=[("timestamp", -1)],
            limit=limit
        )
    
    async def find_errors(
        self,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        hours_back: int = 24,
        limit: int = 100
    ) -> List[RawLogDocument]:
        """Find error logs within specified time range."""
        filter_dict = {
            "log_level": {"$in": ["ERROR", "CRITICAL"]},
            "timestamp": {"$gte": datetime.utcnow() - timedelta(hours=hours_back)}
        }
        
        if agent_id:
            filter_dict["agent_id"] = agent_id
        if session_id:
            filter_dict["session_id"] = session_id
        
        return await self.find_many(
            filter_dict,
            sort=[("timestamp", -1)],
            limit=limit
        )
    
    async def search_logs(
        self,
        search_text: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        framework: Optional[str] = None,
        log_level: Optional[str] = None,
        limit: int = 100
    ) -> List[RawLogDocument]:
        """Search logs using text search."""
        filter_dict: Dict[str, Any] = {}
        
        if agent_id:
            filter_dict["agent_id"] = agent_id
        if session_id:
            filter_dict["session_id"] = session_id
        if framework:
            filter_dict["framework"] = framework.lower()
        if log_level:
            filter_dict["log_level"] = log_level.upper()
        
        return await self.text_search(search_text, filter_dict, limit)
    
    async def get_log_statistics(
        self,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        hours_back: int = 24
    ) -> Dict[str, Any]:
        """Get log statistics for analysis."""
        match_stage: Dict[str, Any] = {
            "timestamp": {"$gte": datetime.utcnow() - timedelta(hours=hours_back)}
        }
        
        if agent_id:
            match_stage["agent_id"] = agent_id
        if session_id:
            match_stage["session_id"] = session_id
        
        pipeline = [
            {"$match": match_stage},
            {
                "$group": {
                    "_id": {
                        "log_level": "$log_level",
                        "framework": "$framework"
                    },
                    "count": {"$sum": 1},
                    "latest_timestamp": {"$max": "$timestamp"}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "by_level": {
                        "$push": {
                            "log_level": "$_id.log_level",
                            "framework": "$_id.framework",
                            "count": "$count",
                            "latest_timestamp": "$latest_timestamp"
                        }
                    },
                    "total_logs": {"$sum": "$count"}
                }
            }
        ]
        
        result = await self.aggregate(pipeline)
        if result:
            return result[0]
        return {"by_level": [], "total_logs": 0}
    
    async def cleanup_old_logs(self, days_to_keep: int = 30) -> int:
        """Manually cleanup old logs beyond retention period."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        return await self.delete_many({"timestamp": {"$lt": cutoff_date}})
    
    async def get_recent_logs_by_pattern(
        self,
        pattern_regex: str,
        hours_back: int = 1,
        limit: int = 50
    ) -> List[RawLogDocument]:
        """Find recent logs matching a regex pattern."""
        filter_dict = {
            "message": {"$regex": pattern_regex, "$options": "i"},
            "timestamp": {"$gte": datetime.utcnow() - timedelta(hours=hours_back)}
        }
        
        return await self.find_many(
            filter_dict,
            sort=[("timestamp", -1)],
            limit=limit
        )