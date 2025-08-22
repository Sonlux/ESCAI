"""
Base repository class for MongoDB operations in ESCAI Framework.
"""

from typing import Dict, List, Any, Optional, Type, TypeVar, Generic
from datetime import datetime
import logging
from bson import ObjectId
from pymongo.errors import DuplicateKeyError, PyMongoError
from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase

from ..mongo_models import MongoBaseModel

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=MongoBaseModel)


class MongoBaseRepository(Generic[T]):
    """Base repository for MongoDB document operations."""
    
    def __init__(self, db: AsyncIOMotorDatabase, collection_name: str, model_class: Type[T]):
        """Initialize repository with database and collection."""
        self.db = db
        self.collection: AsyncIOMotorCollection = db[collection_name]
        self.model_class = model_class
        self.collection_name = collection_name
    
    async def create_indexes(self):
        """Create indexes for the collection. Override in subclasses."""
        # Create default indexes
        await self.collection.create_index("created_at")
        await self.collection.create_index("updated_at")
        logger.info(f"Created default indexes for {self.collection_name}")
    
    async def insert_one(self, document: T) -> str:
        """Insert a single document."""
        try:
            doc_dict = document.dict(by_alias=True, exclude_unset=True)
            if "_id" not in doc_dict:
                doc_dict["_id"] = ObjectId()
            
            result = await self.collection.insert_one(doc_dict)
            logger.debug(f"Inserted document with ID: {result.inserted_id}")
            return str(result.inserted_id)
        
        except DuplicateKeyError as e:
            logger.error(f"Duplicate key error inserting document: {e}")
            raise
        except PyMongoError as e:
            logger.error(f"MongoDB error inserting document: {e}")
            raise
    
    async def insert_many(self, documents: List[T]) -> List[str]:
        """Insert multiple documents."""
        try:
            doc_dicts = []
            for doc in documents:
                doc_dict = doc.dict(by_alias=True, exclude_unset=True)
                if "_id" not in doc_dict:
                    doc_dict["_id"] = ObjectId()
                doc_dicts.append(doc_dict)
            
            result = await self.collection.insert_many(doc_dicts)
            inserted_ids = [str(id_) for id_ in result.inserted_ids]
            logger.debug(f"Inserted {len(inserted_ids)} documents")
            return inserted_ids
        
        except PyMongoError as e:
            logger.error(f"MongoDB error inserting documents: {e}")
            raise
    
    async def find_by_id(self, document_id: str) -> Optional[T]:
        """Find a document by ID."""
        try:
            if not ObjectId.is_valid(document_id):
                return None
            
            doc = await self.collection.find_one({"_id": ObjectId(document_id)})
            if doc:
                return self.model_class(**doc)
            return None
        
        except PyMongoError as e:
            logger.error(f"MongoDB error finding document by ID: {e}")
            raise
    
    async def find_one(self, filter_dict: Dict[str, Any]) -> Optional[T]:
        """Find a single document by filter."""
        try:
            doc = await self.collection.find_one(filter_dict)
            if doc:
                return self.model_class(**doc)
            return None
        
        except PyMongoError as e:
            logger.error(f"MongoDB error finding document: {e}")
            raise
    
    async def find_many(
        self,
        filter_dict: Dict[str, Any] = None,
        sort: List[tuple] = None,
        limit: int = None,
        skip: int = None
    ) -> List[T]:
        """Find multiple documents by filter."""
        try:
            if filter_dict is None:
                filter_dict = {}
            
            cursor = self.collection.find(filter_dict)
            
            if sort:
                cursor = cursor.sort(sort)
            if skip:
                cursor = cursor.skip(skip)
            if limit:
                cursor = cursor.limit(limit)
            
            docs = await cursor.to_list(length=limit)
            return [self.model_class(**doc) for doc in docs]
        
        except PyMongoError as e:
            logger.error(f"MongoDB error finding documents: {e}")
            raise
    
    async def update_one(
        self,
        filter_dict: Dict[str, Any],
        update_dict: Dict[str, Any],
        upsert: bool = False
    ) -> bool:
        """Update a single document."""
        try:
            # Add updated_at timestamp
            if "$set" not in update_dict:
                update_dict["$set"] = {}
            update_dict["$set"]["updated_at"] = datetime.utcnow()
            
            result = await self.collection.update_one(filter_dict, update_dict, upsert=upsert)
            return result.modified_count > 0 or (upsert and result.upserted_id is not None)
        
        except PyMongoError as e:
            logger.error(f"MongoDB error updating document: {e}")
            raise
    
    async def update_by_id(self, document_id: str, update_dict: Dict[str, Any]) -> bool:
        """Update a document by ID."""
        if not ObjectId.is_valid(document_id):
            return False
        
        return await self.update_one({"_id": ObjectId(document_id)}, update_dict)
    
    async def update_many(
        self,
        filter_dict: Dict[str, Any],
        update_dict: Dict[str, Any]
    ) -> int:
        """Update multiple documents."""
        try:
            # Add updated_at timestamp
            if "$set" not in update_dict:
                update_dict["$set"] = {}
            update_dict["$set"]["updated_at"] = datetime.utcnow()
            
            result = await self.collection.update_many(filter_dict, update_dict)
            return result.modified_count
        
        except PyMongoError as e:
            logger.error(f"MongoDB error updating documents: {e}")
            raise
    
    async def delete_one(self, filter_dict: Dict[str, Any]) -> bool:
        """Delete a single document."""
        try:
            result = await self.collection.delete_one(filter_dict)
            return result.deleted_count > 0
        
        except PyMongoError as e:
            logger.error(f"MongoDB error deleting document: {e}")
            raise
    
    async def delete_by_id(self, document_id: str) -> bool:
        """Delete a document by ID."""
        if not ObjectId.is_valid(document_id):
            return False
        
        return await self.delete_one({"_id": ObjectId(document_id)})
    
    async def delete_many(self, filter_dict: Dict[str, Any]) -> int:
        """Delete multiple documents."""
        try:
            result = await self.collection.delete_many(filter_dict)
            return result.deleted_count
        
        except PyMongoError as e:
            logger.error(f"MongoDB error deleting documents: {e}")
            raise
    
    async def count_documents(self, filter_dict: Dict[str, Any] = None) -> int:
        """Count documents matching filter."""
        try:
            if filter_dict is None:
                filter_dict = {}
            return await self.collection.count_documents(filter_dict)
        
        except PyMongoError as e:
            logger.error(f"MongoDB error counting documents: {e}")
            raise
    
    async def aggregate(self, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute aggregation pipeline."""
        try:
            cursor = self.collection.aggregate(pipeline)
            return await cursor.to_list(length=None)
        
        except PyMongoError as e:
            logger.error(f"MongoDB error executing aggregation: {e}")
            raise
    
    async def create_text_index(self, fields: List[str]):
        """Create text index for full-text search."""
        try:
            index_spec = [(field, "text") for field in fields]
            await self.collection.create_index(index_spec)
            logger.info(f"Created text index on fields: {fields}")
        
        except PyMongoError as e:
            logger.error(f"MongoDB error creating text index: {e}")
            raise
    
    async def text_search(
        self,
        search_text: str,
        filter_dict: Dict[str, Any] = None,
        limit: int = None
    ) -> List[T]:
        """Perform text search."""
        try:
            query = {"$text": {"$search": search_text}}
            if filter_dict:
                query.update(filter_dict)
            
            cursor = self.collection.find(query)
            if limit:
                cursor = cursor.limit(limit)
            
            docs = await cursor.to_list(length=limit)
            return [self.model_class(**doc) for doc in docs]
        
        except PyMongoError as e:
            logger.error(f"MongoDB error performing text search: {e}")
            raise