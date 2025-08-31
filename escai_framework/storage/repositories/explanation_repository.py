"""
Repository for generated explanations in MongoDB.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
from motor.motor_asyncio import AsyncIOMotorDatabase

from .mongo_base_repository import MongoBaseRepository
from ..mongo_models import ExplanationDocument

logger = logging.getLogger(__name__)


class ExplanationRepository(MongoBaseRepository[ExplanationDocument]):
    """Repository for managing generated explanations."""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db, "explanations", ExplanationDocument)
    
    async def create_indexes(self):
        """Create indexes optimized for explanation queries."""
        await super().create_indexes()
        
        # Compound indexes for common queries
        await self.collection.create_index([
            ("agent_id", 1),
            ("created_at", -1)
        ])
        
        await self.collection.create_index([
            ("session_id", 1),
            ("created_at", -1)
        ])
        
        await self.collection.create_index([
            ("explanation_type", 1),
            ("created_at", -1)
        ])
        
        await self.collection.create_index([
            ("agent_id", 1),
            ("explanation_type", 1),
            ("created_at", -1)
        ])
        
        # Index for confidence-based queries
        await self.collection.create_index([
            ("confidence_score", -1),
            ("created_at", -1)
        ])
        
        # Text index for content search
        await self.create_text_index(["title", "content"])
        
        # TTL index for automatic cleanup (180 days)
        await self.collection.create_index(
            "created_at",
            expireAfterSeconds=180 * 24 * 60 * 60  # 180 days
        )
        
        logger.info("Created indexes for explanations collection")
    
    async def find_by_agent(
        self,
        agent_id: str,
        explanation_type: Optional[str] = None,
        min_confidence: Optional[float] = None,
        limit: int = 100
    ) -> List[ExplanationDocument]:
        """Find explanations for a specific agent."""
        filter_dict: Dict[str, Any] = {"agent_id": agent_id}
        
        if explanation_type:
            filter_dict["explanation_type"] = explanation_type
        
        if min_confidence is not None:
            filter_dict["confidence_score"] = {"$gte": min_confidence}
        
        return await self.find_many(
            filter_dict,
            sort=[("created_at", -1)],
            limit=limit
        )
    
    async def find_by_session(
        self,
        session_id: str,
        explanation_type: Optional[str] = None,
        limit: int = 100
    ) -> List[ExplanationDocument]:
        """Find explanations for a specific monitoring session."""
        filter_dict: Dict[str, Any] = {"session_id": session_id}
        
        if explanation_type:
            filter_dict["explanation_type"] = explanation_type
        
        return await self.find_many(
            filter_dict,
            sort=[("created_at", -1)],
            limit=limit
        )
    
    async def find_by_type(
        self,
        explanation_type: str,
        min_confidence: Optional[float] = None,
        hours_back: Optional[int] = None,
        limit: int = 100
    ) -> List[ExplanationDocument]:
        """Find explanations by type."""
        filter_dict: Dict[str, Any] = {"explanation_type": explanation_type}
        
        if min_confidence is not None:
            filter_dict["confidence_score"] = {"$gte": min_confidence}
        
        if hours_back is not None:
            filter_dict["created_at"] = {
                "$gte": datetime.utcnow() - timedelta(hours=hours_back)
            }
        
        return await self.find_many(
            filter_dict,
            sort=[("confidence_score", -1), ("created_at", -1)],
            limit=limit
        )
    
    async def find_high_confidence_explanations(
        self,
        min_confidence: float = 0.8,
        agent_id: Optional[str] = None,
        hours_back: int = 24,
        limit: int = 50
    ) -> List[ExplanationDocument]:
        """Find high-confidence explanations."""
        filter_dict = {
            "confidence_score": {"$gte": min_confidence},
            "created_at": {"$gte": datetime.utcnow() - timedelta(hours=hours_back)}
        }
        
        if agent_id:
            filter_dict["agent_id"] = agent_id
        
        return await self.find_many(
            filter_dict,
            sort=[("confidence_score", -1), ("created_at", -1)],
            limit=limit
        )
    
    async def find_behavior_summaries(
        self,
        agent_id: str,
        days_back: int = 7,
        limit: int = 20
    ) -> List[ExplanationDocument]:
        """Find behavior summary explanations for an agent."""
        filter_dict = {
            "agent_id": agent_id,
            "explanation_type": "behavior_summary",
            "created_at": {"$gte": datetime.utcnow() - timedelta(days=days_back)}
        }
        
        return await self.find_many(
            filter_dict,
            sort=[("created_at", -1)],
            limit=limit
        )
    
    async def find_failure_analyses(
        self,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        days_back: int = 7,
        limit: int = 20
    ) -> List[ExplanationDocument]:
        """Find failure analysis explanations."""
        filter_dict = {
            "explanation_type": "failure_analysis",
            "created_at": {"$gte": datetime.utcnow() - timedelta(days=days_back)}
        }
        
        if agent_id:
            filter_dict["agent_id"] = agent_id
        if session_id:
            filter_dict["session_id"] = session_id
        
        return await self.find_many(
            filter_dict,
            sort=[("created_at", -1)],
            limit=limit
        )
    
    async def search_explanations(
        self,
        search_text: str,
        agent_id: Optional[str] = None,
        explanation_type: Optional[str] = None,
        min_confidence: Optional[float] = None,
        limit: int = 50
    ) -> List[ExplanationDocument]:
        """Search explanations using text search."""
        filter_dict: Dict[str, Any] = {}
        
        if agent_id:
            filter_dict["agent_id"] = agent_id
        if explanation_type:
            filter_dict["explanation_type"] = explanation_type
        if min_confidence is not None:
            filter_dict["confidence_score"] = {"$gte": min_confidence}
        
        return await self.text_search(search_text, filter_dict, limit)
    
    async def find_related_explanations(
        self,
        event_ids: List[str],
        limit: int = 20
    ) -> List[ExplanationDocument]:
        """Find explanations related to specific events."""
        filter_dict = {"related_events": {"$in": event_ids}}
        
        return await self.find_many(
            filter_dict,
            sort=[("confidence_score", -1), ("created_at", -1)],
            limit=limit
        )
    
    async def get_explanation_statistics(
        self,
        agent_id: Optional[str] = None,
        days_back: int = 7
    ) -> Dict[str, Any]:
        """Get explanation statistics."""
        match_stage: Dict[str, Any] = {
            "created_at": {"$gte": datetime.utcnow() - timedelta(days=days_back)}
        }
        
        if agent_id:
            match_stage["agent_id"] = agent_id
        
        pipeline: List[Dict[str, Any]] = [
            {"$match": match_stage},
            {
                "$group": {
                    "_id": "$explanation_type",
                    "count": {"$sum": 1},
                    "avg_confidence": {"$avg": "$confidence_score"},
                    "max_confidence": {"$max": "$confidence_score"},
                    "min_confidence": {"$min": "$confidence_score"},
                    "latest_created": {"$max": "$created_at"}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "by_type": {
                        "$push": {
                            "explanation_type": "$_id",
                            "count": "$count",
                            "avg_confidence": "$avg_confidence",
                            "max_confidence": "$max_confidence",
                            "min_confidence": "$min_confidence",
                            "latest_created": "$latest_created"
                        }
                    },
                    "total_explanations": {"$sum": "$count"},
                    "overall_avg_confidence": {"$avg": "$avg_confidence"}
                }
            }
        ]
        
        result = await self.aggregate(pipeline)
        if result:
            return result[0]
        return {"by_type": [], "total_explanations": 0, "overall_avg_confidence": 0.0}
    
    async def get_confidence_distribution(
        self,
        agent_id: Optional[str] = None,
        explanation_type: Optional[str] = None,
        days_back: int = 7
    ) -> List[Dict[str, Any]]:
        """Get confidence score distribution."""
        match_stage: Dict[str, Any] = {
            "created_at": {"$gte": datetime.utcnow() - timedelta(days=days_back)}
        }
        
        if agent_id:
            match_stage["agent_id"] = agent_id
        if explanation_type:
            match_stage["explanation_type"] = explanation_type
        
        pipeline: List[Dict[str, Any]] = [
            {"$match": match_stage},
            {
                "$bucket": {
                    "groupBy": "$confidence_score",
                    "boundaries": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    "default": "other",
                    "output": {
                        "count": {"$sum": 1},
                        "avg_confidence": {"$avg": "$confidence_score"}
                    }
                }
            }
        ]
        
        return await self.aggregate(pipeline)
    
    async def cleanup_old_explanations(self, days_to_keep: int = 180) -> int:
        """Manually cleanup old explanations beyond retention period."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        return await self.delete_many({"created_at": {"$lt": cutoff_date}})