"""
Caching system for frequently accessed data in CLI operations.
Implements intelligent caching with TTL, LRU eviction, and memory management.
"""

import time
import threading
import pickle
import hashlib
import os
from typing import Any, Dict, Optional, Callable, Union, List
from dataclasses import dataclass, field
from collections import OrderedDict
from abc import ABC, abstractmethod
import weakref
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for cache behavior."""
    max_size: int = 100  # Maximum number of items
    max_memory_mb: int = 50  # Maximum memory usage in MB
    default_ttl: int = 300  # Default TTL in seconds (5 minutes)
    cleanup_interval: int = 60  # Cleanup interval in seconds
    enable_persistence: bool = True  # Enable disk persistence
    cache_dir: str = field(default_factory=lambda: os.path.expanduser("~/.escai/cache"))


class CacheItem:
    """Represents a cached item with metadata."""
    
    def __init__(self, key: str, value: Any, ttl: Optional[int] = None, size_estimate: int = 0):
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.last_accessed = self.created_at
        self.access_count = 0
        self.ttl = ttl
        self.size_estimate = size_estimate
        self._lock = threading.Lock()
    
    def is_expired(self) -> bool:
        """Check if the item has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def access(self) -> Any:
        """Access the cached value and update metadata."""
        with self._lock:
            self.last_accessed = time.time()
            self.access_count += 1
            return self.value
    
    def get_age(self) -> float:
        """Get the age of the item in seconds."""
        return time.time() - self.created_at
    
    def get_priority_score(self) -> float:
        """Calculate priority score for eviction (higher = keep longer)."""
        age_factor = 1.0 / (1.0 + self.get_age() / 3600)  # Decay over hours
        access_factor = min(self.access_count / 10.0, 1.0)  # Cap at 10 accesses
        recency_factor = 1.0 / (1.0 + (time.time() - self.last_accessed) / 3600)
        
        return age_factor * 0.3 + access_factor * 0.4 + recency_factor * 0.3


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[CacheItem]:
        pass
    
    @abstractmethod
    def put(self, key: str, item: CacheItem) -> bool:
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        pass
    
    @abstractmethod
    def clear(self) -> None:
        pass
    
    @abstractmethod
    def keys(self) -> List[str]:
        pass


class MemoryCache(CacheBackend):
    """In-memory cache backend with LRU eviction."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache: OrderedDict[str, CacheItem] = OrderedDict()
        self._lock = threading.RLock()
        self._memory_usage = 0
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def get(self, key: str) -> Optional[CacheItem]:
        """Get item from cache."""
        with self._lock:
            if key not in self.cache:
                return None
            
            item = self.cache[key]
            
            # Check expiration
            if item.is_expired():
                del self.cache[key]
                self._memory_usage -= item.size_estimate
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return item
    
    def put(self, key: str, item: CacheItem) -> bool:
        """Put item in cache."""
        with self._lock:
            # Remove existing item if present
            if key in self.cache:
                old_item = self.cache[key]
                self._memory_usage -= old_item.size_estimate
                del self.cache[key]
            
            # Check if we need to evict items
            self._evict_if_needed(item.size_estimate)
            
            # Add new item
            self.cache[key] = item
            self._memory_usage += item.size_estimate
            
            logger.debug(f"Cached item {key}, memory usage: {self._memory_usage / (1024*1024):.2f}MB")
            return True
    
    def delete(self, key: str) -> bool:
        """Delete item from cache."""
        with self._lock:
            if key in self.cache:
                item = self.cache[key]
                self._memory_usage -= item.size_estimate
                del self.cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all items from cache."""
        with self._lock:
            self.cache.clear()
            self._memory_usage = 0
    
    def keys(self) -> List[str]:
        """Get all cache keys."""
        with self._lock:
            return list(self.cache.keys())
    
    def _evict_if_needed(self, new_item_size: int):
        """Evict items if cache limits would be exceeded."""
        max_memory_bytes = self.config.max_memory_mb * 1024 * 1024
        
        # Evict by size limit
        while (len(self.cache) >= self.config.max_size or 
               self._memory_usage + new_item_size > max_memory_bytes):
            if not self.cache:
                break
            
            # Find item with lowest priority score
            worst_key = min(self.cache.keys(), 
                          key=lambda k: self.cache[k].get_priority_score())
            
            worst_item = self.cache[worst_key]
            self._memory_usage -= worst_item.size_estimate
            del self.cache[worst_key]
            
            logger.debug(f"Evicted cache item {worst_key}")
    
    def _cleanup_worker(self):
        """Background worker to clean up expired items."""
        while True:
            try:
                time.sleep(self.config.cleanup_interval)
                self._cleanup_expired()
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    def _cleanup_expired(self):
        """Remove expired items from cache."""
        with self._lock:
            expired_keys = [key for key, item in self.cache.items() if item.is_expired()]
            
            for key in expired_keys:
                item = self.cache[key]
                self._memory_usage -= item.size_estimate
                del self.cache[key]
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache items")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self.cache),
                "max_size": self.config.max_size,
                "memory_usage_mb": self._memory_usage / (1024 * 1024),
                "max_memory_mb": self.config.max_memory_mb,
                "hit_ratio": self._calculate_hit_ratio()
            }
    
    def _calculate_hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        if not self.cache:
            return 0.0
        
        total_accesses = sum(item.access_count for item in self.cache.values())
        return min(total_accesses / len(self.cache), 1.0) if total_accesses > 0 else 0.0


class PersistentCache(CacheBackend):
    """Persistent cache backend using disk storage."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.memory_cache = MemoryCache(config)
        
        # Ensure cache directory exists
        os.makedirs(config.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(config.cache_dir, "cli_cache.pkl")
        
        # Load existing cache
        self._load_from_disk()
    
    def get(self, key: str) -> Optional[CacheItem]:
        """Get item from cache (memory first, then disk)."""
        # Try memory cache first
        item = self.memory_cache.get(key)
        if item:
            return item
        
        # Try loading from disk
        return self._load_from_disk_key(key)
    
    def put(self, key: str, item: CacheItem) -> bool:
        """Put item in cache (memory and disk)."""
        success = self.memory_cache.put(key, item)
        if success and self.config.enable_persistence:
            self._save_to_disk_key(key, item)
        return success
    
    def delete(self, key: str) -> bool:
        """Delete item from cache."""
        memory_deleted = self.memory_cache.delete(key)
        disk_deleted = self._delete_from_disk_key(key)
        return memory_deleted or disk_deleted
    
    def clear(self) -> None:
        """Clear all items from cache."""
        self.memory_cache.clear()
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
    
    def keys(self) -> List[str]:
        """Get all cache keys."""
        memory_keys = set(self.memory_cache.keys())
        disk_keys = set(self._get_disk_keys())
        return list(memory_keys.union(disk_keys))
    
    def _load_from_disk(self):
        """Load cache from disk."""
        if not os.path.exists(self.cache_file):
            return
        
        try:
            with open(self.cache_file, 'rb') as f:
                disk_cache = pickle.load(f)
                
            # Load non-expired items into memory cache
            for key, item in disk_cache.items():
                if not item.is_expired():
                    self.memory_cache.put(key, item)
                    
        except Exception as e:
            logger.error(f"Failed to load cache from disk: {e}")
    
    def _save_to_disk_key(self, key: str, item: CacheItem):
        """Save a single item to disk."""
        try:
            # Load existing cache
            disk_cache = {}
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    disk_cache = pickle.load(f)
            
            # Update with new item
            disk_cache[key] = item
            
            # Save back to disk
            with open(self.cache_file, 'wb') as f:
                pickle.dump(disk_cache, f)
                
        except Exception as e:
            logger.error(f"Failed to save cache item to disk: {e}")
    
    def _load_from_disk_key(self, key: str) -> Optional[CacheItem]:
        """Load a single item from disk."""
        if not os.path.exists(self.cache_file):
            return None
        
        try:
            with open(self.cache_file, 'rb') as f:
                disk_cache = pickle.load(f)
            
            if key in disk_cache:
                item = disk_cache[key]
                if not item.is_expired():
                    # Also add to memory cache
                    self.memory_cache.put(key, item)
                    return item
                    
        except Exception as e:
            logger.error(f"Failed to load cache item from disk: {e}")
        
        return None
    
    def _delete_from_disk_key(self, key: str) -> bool:
        """Delete a single item from disk."""
        if not os.path.exists(self.cache_file):
            return False
        
        try:
            with open(self.cache_file, 'rb') as f:
                disk_cache = pickle.load(f)
            
            if key in disk_cache:
                del disk_cache[key]
                
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(disk_cache, f)
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete cache item from disk: {e}")
        
        return False
    
    def _get_disk_keys(self) -> List[str]:
        """Get all keys from disk cache."""
        if not os.path.exists(self.cache_file):
            return []
        
        try:
            with open(self.cache_file, 'rb') as f:
                disk_cache = pickle.load(f)
            return list(disk_cache.keys())
        except Exception as e:
            logger.error(f"Failed to get disk cache keys: {e}")
            return []


class CLICacheManager:
    """Main cache manager for CLI operations."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        
        if self.config.enable_persistence:
            self.backend = PersistentCache(self.config)
        else:
            self.backend = MemoryCache(self.config)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        item = self.backend.get(key)
        if item:
            return item.access()
        return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Put value in cache."""
        if ttl is None:
            ttl = self.config.default_ttl
        
        # Estimate size
        size_estimate = self._estimate_size(value)
        
        item = CacheItem(key, value, ttl, size_estimate)
        return self.backend.put(key, item)
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        return self.backend.delete(key)
    
    def clear(self) -> None:
        """Clear all cached values."""
        self.backend.clear()
    
    def cached(self, ttl: Optional[int] = None, key_func: Optional[Callable] = None):
        """Decorator for caching function results."""
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self._generate_key(func.__name__, args, kwargs)
                
                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for {cache_key}")
                    return cached_result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.put(cache_key, result, ttl)
                logger.debug(f"Cached result for {cache_key}")
                
                return result
            
            return wrapper
        return decorator
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate a cache key from function name and arguments."""
        key_data = {
            'func': func_name,
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        
        key_str = str(key_data)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate the size of a value in bytes."""
        try:
            return len(pickle.dumps(value))
        except Exception:
            # Fallback estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (list, tuple)):
                return sum(self._estimate_size(item) for item in value[:10])  # Sample first 10
            elif isinstance(value, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) 
                          for k, v in list(value.items())[:10])  # Sample first 10
            else:
                return 1000  # Default estimate
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if hasattr(self.backend, 'get_stats'):
            return self.backend.get_stats()
        else:
            return {"backend": type(self.backend).__name__}


# Global cache instance for CLI
_cli_cache: Optional[CLICacheManager] = None


def get_cli_cache() -> CLICacheManager:
    """Get the global CLI cache instance."""
    global _cli_cache
    if _cli_cache is None:
        _cli_cache = CLICacheManager()
    return _cli_cache


def cache_analysis_result(ttl: int = 600):
    """Decorator for caching analysis results."""
    return get_cli_cache().cached(ttl=ttl)


def cache_monitoring_data(ttl: int = 300):
    """Decorator for caching monitoring data."""
    return get_cli_cache().cached(ttl=ttl)


def clear_cli_cache():
    """Clear the CLI cache."""
    get_cli_cache().clear()