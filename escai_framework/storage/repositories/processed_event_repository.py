"""
Repository for processed agent events in MongoDB.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
from motor.motor_asyncio import AsyncIOMotorDatabase

from .mongo_base_repository import MongoBaseRepository
from ..mongo_models import ProcessedEventDocument

logger = logging.getLogger(__name__)


class ProcessedEventRepository(MongoBaseRepository[ProcessedEventDocument]):
    """Repository for managing processed agent events."""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db, "processed_events", ProcessedEventDocument)
    
    async def create_indexes(self):
        """Create indexes optimized for event queries."""
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
            ("event_type", 1),
            ("timestamp", -1)
        ])
        
        await self.collection.create_index([
            ("agent_id", 1),
            ("event_type", 1),
            ("timestamp", -1)
        ])
        
        # Index for processing metadata queries
        await self.collection.create_index("processed_at")
        
        # TTL index for automatic cleanup (90 days)
        await self.collection.create_index(
            "timestamp",
            expireAfterSeconds=90 * 24 * 60 * 60  # 90 days
        )
        
        logger.info("Created indexes for processed_events collection")
    
    async def find_by_agent(
        self,
        agent_id: str,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[ProcessedEventDocument]:
        """Find events for a specific agent."""
        filter_dict: Dict[str, Any] = {"agent_id": agent_id}
        
        if event_type:
            filter_dict["event_type"] = event_type
        
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
    
    async def find_by_session(
        self,
        session_id: str,
        event_type: Optional[str] = None,
        limit: int = 1000
    ) -> List[ProcessedEventDocument]:
        """Find events for a specific monitoring session."""
        filter_dict: Dict[str, Any] = {"session_id": session_id}
        
        if event_type:
            filter_dict["event_type"] = event_type
        
        return await self.find_many(
            filter_dict,
            sort=[("timestamp", -1)],
            limit=limit
        )
    
    async def find_by_event_type(
        self,
        event_type: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[ProcessedEventDocument]:
        """Find events by type."""
        filter_dict: Dict[str, Any] = {"event_type": event_type}
        
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
    
    async def find_decision_events(
        self,
        agent_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[ProcessedEventDocument]:
        """Find decision-making events for an agent."""
        return await self.find_by_agent(
            agent_id,
            event_type="decision_made",
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
    
    async def find_tool_usage_events(
        self,
        agent_id: str,
        tool_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[ProcessedEventDocument]:
        """Find tool usage events for an agent."""
        filter_dict: Dict[str, Any] = {
            "agent_id": agent_id,
            "event_type": "tool_used"
        }
        
        if tool_name:
            filter_dict["event_data.tool_name"] = tool_name
        
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
    
    async def find_error_events(
        self,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        hours_back: int = 24,
        limit: int = 100
    ) -> List[ProcessedEventDocument]:
        """Find error events within specified time range."""
        filter_dict = {
            "event_type": "error_occurred",
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
    
    async def find_pattern_events(
        self,
        agent_id: Optional[str] = None,
        pattern_type: Optional[str] = None,
        hours_back: int = 24,
        limit: int = 100
    ) -> List[ProcessedEventDocument]:
        """Find pattern detection events."""
        filter_dict = {
            "event_type": {"$in": ["pattern_detected", "anomaly_detected"]},
            "timestamp": {"$gte": datetime.utcnow() - timedelta(hours=hours_back)}
        }
        
        if agent_id:
            filter_dict["agent_id"] = agent_id
        if pattern_type:
            filter_dict["event_data.pattern_type"] = pattern_type
        
        return await self.find_many(
            filter_dict,
            sort=[("timestamp", -1)],
            limit=limit
        )
    
    async def get_event_statistics(
        self,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        hours_back: int = 24
    ) -> Dict[str, Any]:
        """Get event statistics for analysis."""
        match_stage: Dict[str, Any] = {
            "timestamp": {"$gte": datetime.utcnow() - timedelta(hours=hours_back)}
        }
        
        if agent_id:
            match_stage["agent_id"] = agent_id
        if session_id:
            match_stage["session_id"] = session_id
        
        pipeline: List[Dict[str, Any]] = [
            {"$match": match_stage},
            {
                "$group": {
                    "_id": "$event_type",
                    "count": {"$sum": 1},
                    "latest_timestamp": {"$max": "$timestamp"},
                    "earliest_timestamp": {"$min": "$timestamp"}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "by_type": {
                        "$push": {
                            "event_type": "$_id",
                            "count": "$count",
                            "latest_timestamp": "$latest_timestamp",
                            "earliest_timestamp": "$earliest_timestamp"
                        }
                    },
                    "total_events": {"$sum": "$count"}
                }
            }
        ]
        
        result = await self.aggregate(pipeline)
        if result:
            return result[0]
        return {"by_type": [], "total_events": 0}
    
    async def get_event_timeline(
        self,
        agent_id: str,
        start_time: datetime,
        end_time: datetime,
        bucket_size_minutes: int = 5
    ) -> List[Dict[str, Any]]:
        """Get event timeline with time buckets."""
        pipeline: List[Dict[str, Any]] = [
            {
                "$match": {
                    "agent_id": agent_id,
                    "timestamp": {"$gte": start_time, "$lte": end_time}
                }
            },
            {
                "$group": {
                    "_id": {
                        "time_bucket": {
                            "$dateTrunc": {
                                "date": "$timestamp",
                                "unit": "minute",
                                "binSize": bucket_size_minutes
                            }
                        },
                        "event_type": "$event_type"
                    },
                    "count": {"$sum": 1}
                }
            },
            {
                "$group": {
                    "_id": "$_id.time_bucket",
                    "events": {
                        "$push": {
                            "event_type": "$_id.event_type",
                            "count": "$count"
                        }
                    },
                    "total_count": {"$sum": "$count"}
                }
            },
            {"$sort": {"_id": 1}}
        ]
        
        return await self.aggregate(pipeline)
    
    async def find_events_by_source_logs(
        self,
        source_log_ids: List[str],
        limit: int = 100
    ) -> List[ProcessedEventDocument]:
        """Find events that were processed from specific log entries."""
        filter_dict = {"source_log_ids": {"$in": source_log_ids}}
        
        return await self.find_many(
            filter_dict,
            sort=[("processed_at", -1)],
            limit=limit
        )
    
    async def cleanup_old_events(self, days_to_keep: int = 90) -> int:
        """Manually cleanup old events beyond retention period."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        return await self.delete_many({"timestamp": {"$lt": cutoff_date}})