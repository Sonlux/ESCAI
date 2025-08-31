"""
Repository for analytics and ML model results in MongoDB.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
from motor.motor_asyncio import AsyncIOMotorDatabase

from .mongo_base_repository import MongoBaseRepository
from ..mongo_models import AnalyticsResultDocument

logger = logging.getLogger(__name__)


class AnalyticsResultRepository(MongoBaseRepository[AnalyticsResultDocument]):
    """Repository for managing analytics and ML model results."""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db, "analytics_results", AnalyticsResultDocument)
    
    async def create_indexes(self):
        """Create indexes optimized for analytics queries."""
        await super().create_indexes()
        
        # Compound indexes for common queries
        await self.collection.create_index([
            ("analysis_type", 1),
            ("created_at", -1)
        ])
        
        await self.collection.create_index([
            ("model_name", 1),
            ("model_version", 1),
            ("created_at", -1)
        ])
        
        await self.collection.create_index([
            ("agent_id", 1),
            ("analysis_type", 1),
            ("created_at", -1)
        ])
        
        await self.collection.create_index([
            ("session_id", 1),
            ("analysis_type", 1),
            ("created_at", -1)
        ])
        
        # Index for input data hash (for reproducibility)
        await self.collection.create_index("input_data_hash")
        
        # Index for execution time queries
        await self.collection.create_index([
            ("execution_time_ms", 1),
            ("analysis_type", 1)
        ])
        
        # TTL index for automatic cleanup (365 days)
        await self.collection.create_index(
            "created_at",
            expireAfterSeconds=365 * 24 * 60 * 60  # 365 days
        )
        
        logger.info("Created indexes for analytics_results collection")
    
    async def find_by_analysis_type(
        self,
        analysis_type: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        hours_back: Optional[int] = None,
        limit: int = 100
    ) -> List[AnalyticsResultDocument]:
        """Find results by analysis type."""
        filter_dict: Dict[str, Any] = {"analysis_type": analysis_type}
        
        if agent_id:
            filter_dict["agent_id"] = agent_id
        if session_id:
            filter_dict["session_id"] = session_id
        
        if hours_back is not None:
            filter_dict["created_at"] = {
                "$gte": datetime.utcnow() - timedelta(hours=hours_back)
            }
        
        return await self.find_many(
            filter_dict,
            sort=[("created_at", -1)],
            limit=limit
        )
    
    async def find_by_model(
        self,
        model_name: str,
        model_version: Optional[str] = None,
        analysis_type: Optional[str] = None,
        limit: int = 100
    ) -> List[AnalyticsResultDocument]:
        """Find results by model name and version."""
        filter_dict = {"model_name": model_name}
        
        if model_version:
            filter_dict["model_version"] = model_version
        if analysis_type:
            filter_dict["analysis_type"] = analysis_type
        
        return await self.find_many(
            filter_dict,
            sort=[("created_at", -1)],
            limit=limit
        )
    
    async def find_by_agent(
        self,
        agent_id: str,
        analysis_type: Optional[str] = None,
        days_back: int = 7,
        limit: int = 100
    ) -> List[AnalyticsResultDocument]:
        """Find results for a specific agent."""
        filter_dict = {
            "agent_id": agent_id,
            "created_at": {"$gte": datetime.utcnow() - timedelta(days=days_back)}
        }
        
        if analysis_type:
            filter_dict["analysis_type"] = analysis_type
        
        return await self.find_many(
            filter_dict,
            sort=[("created_at", -1)],
            limit=limit
        )
    
    async def find_by_session(
        self,
        session_id: str,
        analysis_type: Optional[str] = None,
        limit: int = 100
    ) -> List[AnalyticsResultDocument]:
        """Find results for a specific session."""
        filter_dict = {"session_id": session_id}
        
        if analysis_type:
            filter_dict["analysis_type"] = analysis_type
        
        return await self.find_many(
            filter_dict,
            sort=[("created_at", -1)],
            limit=limit
        )
    
    async def find_by_input_hash(
        self,
        input_data_hash: str
    ) -> List[AnalyticsResultDocument]:
        """Find results with the same input data hash (for reproducibility)."""
        filter_dict = {"input_data_hash": input_data_hash}
        
        return await self.find_many(
            filter_dict,
            sort=[("created_at", -1)]
        )
    
    async def find_pattern_mining_results(
        self,
        agent_id: Optional[str] = None,
        days_back: int = 7,
        limit: int = 50
    ) -> List[AnalyticsResultDocument]:
        """Find pattern mining results."""
        return await self.find_by_analysis_type(
            "pattern_mining",
            agent_id=agent_id,
            hours_back=days_back * 24,
            limit=limit
        )
    
    async def find_anomaly_detection_results(
        self,
        agent_id: Optional[str] = None,
        days_back: int = 7,
        limit: int = 50
    ) -> List[AnalyticsResultDocument]:
        """Find anomaly detection results."""
        return await self.find_by_analysis_type(
            "anomaly_detection",
            agent_id=agent_id,
            hours_back=days_back * 24,
            limit=limit
        )
    
    async def find_prediction_results(
        self,
        agent_id: Optional[str] = None,
        days_back: int = 7,
        limit: int = 50
    ) -> List[AnalyticsResultDocument]:
        """Find performance prediction results."""
        return await self.find_by_analysis_type(
            "performance_prediction",
            agent_id=agent_id,
            hours_back=days_back * 24,
            limit=limit
        )
    
    async def find_causal_inference_results(
        self,
        agent_id: Optional[str] = None,
        days_back: int = 7,
        limit: int = 50
    ) -> List[AnalyticsResultDocument]:
        """Find causal inference results."""
        return await self.find_by_analysis_type(
            "causal_inference",
            agent_id=agent_id,
            hours_back=days_back * 24,
            limit=limit
        )
    
    async def get_model_performance_metrics(
        self,
        model_name: str,
        model_version: Optional[str] = None,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """Get performance metrics for a specific model."""
        match_stage = {
            "model_name": model_name,
            "created_at": {"$gte": datetime.utcnow() - timedelta(days=days_back)}
        }
        
        if model_version:
            match_stage["model_version"] = model_version
        
        pipeline: List[Dict[str, Any]] = [
            {"$match": match_stage},
            {
                "$group": {
                    "_id": {
                        "model_version": "$model_version",
                        "analysis_type": "$analysis_type"
                    },
                    "count": {"$sum": 1},
                    "avg_execution_time": {"$avg": "$execution_time_ms"},
                    "min_execution_time": {"$min": "$execution_time_ms"},
                    "max_execution_time": {"$max": "$execution_time_ms"},
                    "latest_run": {"$max": "$created_at"}
                }
            },
            {
                "$group": {
                    "_id": "$_id.model_version",
                    "by_analysis_type": {
                        "$push": {
                            "analysis_type": "$_id.analysis_type",
                            "count": "$count",
                            "avg_execution_time": "$avg_execution_time",
                            "min_execution_time": "$min_execution_time",
                            "max_execution_time": "$max_execution_time",
                            "latest_run": "$latest_run"
                        }
                    },
                    "total_runs": {"$sum": "$count"},
                    "overall_avg_execution_time": {"$avg": "$avg_execution_time"}
                }
            }
        ]
        
        result = await self.aggregate(pipeline)
        return result[0] if result else {}
    
    async def get_analysis_statistics(
        self,
        analysis_type: Optional[str] = None,
        days_back: int = 7
    ) -> Dict[str, Any]:
        """Get analysis statistics."""
        match_stage: Dict[str, Any] = {
            "created_at": {"$gte": datetime.utcnow() - timedelta(days=days_back)}
        }
        
        if analysis_type:
            match_stage["analysis_type"] = analysis_type
        
        pipeline: List[Dict[str, Any]] = [
            {"$match": match_stage},
            {
                "$group": {
                    "_id": "$analysis_type",
                    "count": {"$sum": 1},
                    "avg_execution_time": {"$avg": "$execution_time_ms"},
                    "min_execution_time": {"$min": "$execution_time_ms"},
                    "max_execution_time": {"$max": "$execution_time_ms"},
                    "unique_models": {"$addToSet": "$model_name"},
                    "latest_run": {"$max": "$created_at"}
                }
            },
            {
                "$project": {
                    "analysis_type": "$_id",
                    "count": 1,
                    "avg_execution_time": 1,
                    "min_execution_time": 1,
                    "max_execution_time": 1,
                    "unique_model_count": {"$size": "$unique_models"},
                    "latest_run": 1
                }
            },
            {"$sort": {"count": -1}}
        ]
        
        result = await self.aggregate(pipeline)
        return result[0] if result else {}
    
    async def get_execution_time_trends(
        self,
        analysis_type: str,
        model_name: Optional[str] = None,
        days_back: int = 30,
        bucket_hours: int = 6
    ) -> List[Dict[str, Any]]:
        """Get execution time trends over time."""
        match_stage = {
            "analysis_type": analysis_type,
            "created_at": {"$gte": datetime.utcnow() - timedelta(days=days_back)}
        }
        
        if model_name:
            match_stage["model_name"] = model_name
        
        pipeline: List[Dict[str, Any]] = [
            {"$match": match_stage},
            {
                "$group": {
                    "_id": {
                        "time_bucket": {
                            "$dateTrunc": {
                                "date": "$created_at",
                                "unit": "hour",
                                "binSize": bucket_hours
                            }
                        }
                    },
                    "count": {"$sum": 1},
                    "avg_execution_time": {"$avg": "$execution_time_ms"},
                    "min_execution_time": {"$min": "$execution_time_ms"},
                    "max_execution_time": {"$max": "$execution_time_ms"}
                }
            },
            {"$sort": {"_id.time_bucket": 1}}
        ]
        
        result = await self.aggregate(pipeline)
        return result
    
    async def find_slow_analyses(
        self,
        threshold_ms: float = 5000.0,
        analysis_type: Optional[str] = None,
        days_back: int = 7,
        limit: int = 50
    ) -> List[AnalyticsResultDocument]:
        """Find analyses that took longer than threshold."""
        filter_dict = {
            "execution_time_ms": {"$gte": threshold_ms},
            "created_at": {"$gte": datetime.utcnow() - timedelta(days=days_back)}
        }
        
        if analysis_type:
            filter_dict["analysis_type"] = analysis_type
        
        return await self.find_many(
            filter_dict,
            sort=[("execution_time_ms", -1)],
            limit=limit
        )
    
    async def cleanup_old_results(self, days_to_keep: int = 365) -> int:
        """Manually cleanup old results beyond retention period."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        return await self.delete_many({"created_at": {"$lt": cutoff_date}})
    
    async def deduplicate_results(
        self,
        analysis_type: Optional[str] = None,
        keep_latest: bool = True
    ) -> int:
        """Remove duplicate results based on input_data_hash."""
        match_stage: Dict[str, Any] = {}
        if analysis_type:
            match_stage["analysis_type"] = analysis_type
        
        pipeline: List[Dict[str, Any]] = [
            {"$match": match_stage},
            {
                "$group": {
                    "_id": "$input_data_hash",
                    "documents": {
                        "$push": {
                            "id": "$_id",
                            "created_at": "$created_at"
                        }
                    },
                    "count": {"$sum": 1}
                }
            },
            {"$match": {"count": {"$gt": 1}}}
        ]
        
        duplicates = await self.aggregate(pipeline)
        
        deleted_count = 0
        for dup_group in duplicates:
            docs = dup_group["documents"]
            # Sort by created_at
            docs.sort(key=lambda x: x["created_at"], reverse=keep_latest)
            
            # Keep the first one (latest if keep_latest=True), delete the rest
            for doc in docs[1:]:
                if await self.delete_by_id(str(doc["id"])):
                    deleted_count += 1
        
        return deleted_count