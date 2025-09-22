"""
Lazy loading system for large datasets in CLI operations.
Implements efficient data loading strategies to minimize memory usage.
"""

import asyncio
from typing import Any, Dict, List, Optional, Iterator, Callable, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import weakref
import threading
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)


@dataclass
class LazyLoadConfig:
    """Configuration for lazy loading behavior."""
    chunk_size: int = 1000
    max_memory_mb: int = 100
    prefetch_chunks: int = 2
    cache_chunks: int = 5
    enable_compression: bool = True


class DataChunk:
    """Represents a chunk of data that can be loaded on demand."""
    
    def __init__(self, chunk_id: str, loader_func: Callable, size_estimate: int = 0):
        self.chunk_id = chunk_id
        self.loader_func = loader_func
        self.size_estimate = size_estimate
        self._data: Optional[Any] = None
        self._loaded = False
        self._lock = threading.Lock()
    
    def load(self) -> Any:
        """Load the chunk data if not already loaded."""
        if not self._loaded:
            with self._lock:
                if not self._loaded:
                    try:
                        self._data = self.loader_func()
                        self._loaded = True
                        logger.debug(f"Loaded chunk {self.chunk_id}")
                    except Exception as e:
                        logger.error(f"Failed to load chunk {self.chunk_id}: {e}")
                        raise
        return self._data
    
    def unload(self):
        """Unload the chunk data to free memory."""
        with self._lock:
            self._data = None
            self._loaded = False
            logger.debug(f"Unloaded chunk {self.chunk_id}")
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded


class LazyDataLoader:
    """Main lazy loading system for CLI data operations."""
    
    def __init__(self, config: Optional[LazyLoadConfig] = None):
        self.config = config or LazyLoadConfig()
        self.chunks: Dict[str, DataChunk] = {}
        self.chunk_order: List[str] = []
        self.loaded_chunks: weakref.WeakSet = weakref.WeakSet()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._memory_usage = 0
        self._lock = threading.Lock()
    
    def register_chunk(self, chunk_id: str, loader_func: Callable, size_estimate: int = 0):
        """Register a data chunk for lazy loading."""
        chunk = DataChunk(chunk_id, loader_func, size_estimate)
        self.chunks[chunk_id] = chunk
        self.chunk_order.append(chunk_id)
        logger.debug(f"Registered chunk {chunk_id} with estimated size {size_estimate}")
    
    def load_chunk(self, chunk_id: str) -> Any:
        """Load a specific chunk and manage memory."""
        if chunk_id not in self.chunks:
            raise ValueError(f"Chunk {chunk_id} not registered")
        
        chunk = self.chunks[chunk_id]
        
        # Check memory usage before loading
        self._manage_memory()
        
        data = chunk.load()
        self._memory_usage += chunk.size_estimate
        
        return data
    
    def get_chunk_iterator(self, chunk_ids: Optional[List[str]] = None) -> Iterator[Any]:
        """Get an iterator over chunks with automatic loading/unloading."""
        target_chunks = chunk_ids or self.chunk_order
        
        for chunk_id in target_chunks:
            try:
                data = self.load_chunk(chunk_id)
                yield data
                
                # Optionally unload after yielding if memory is tight
                if self._memory_usage > self.config.max_memory_mb * 1024 * 1024:
                    self.chunks[chunk_id].unload()
                    self._memory_usage -= self.chunks[chunk_id].size_estimate
                    
            except Exception as e:
                logger.error(f"Error loading chunk {chunk_id}: {e}")
                continue
    
    def prefetch_chunks(self, chunk_ids: List[str]):
        """Prefetch chunks in background for better performance."""
        def prefetch_worker(chunk_id: str):
            try:
                self.load_chunk(chunk_id)
            except Exception as e:
                logger.error(f"Prefetch failed for chunk {chunk_id}: {e}")
        
        for chunk_id in chunk_ids[:self.config.prefetch_chunks]:
            if chunk_id in self.chunks and not self.chunks[chunk_id].is_loaded:
                self.executor.submit(prefetch_worker, chunk_id)
    
    def _manage_memory(self):
        """Manage memory usage by unloading old chunks."""
        if self._memory_usage > self.config.max_memory_mb * 1024 * 1024:
            # Unload oldest chunks first
            unloaded_count = 0
            for chunk_id in self.chunk_order:
                chunk = self.chunks[chunk_id]
                if chunk.is_loaded and unloaded_count < 2:
                    chunk.unload()
                    self._memory_usage -= chunk.size_estimate
                    unloaded_count += 1
                    
                if self._memory_usage <= self.config.max_memory_mb * 0.8 * 1024 * 1024:
                    break
    
    def clear_all(self):
        """Clear all loaded chunks and reset memory usage."""
        for chunk in self.chunks.values():
            chunk.unload()
        self._memory_usage = 0
        logger.info("Cleared all loaded chunks")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        loaded_count = sum(1 for chunk in self.chunks.values() if chunk.is_loaded)
        return {
            "total_chunks": len(self.chunks),
            "loaded_chunks": loaded_count,
            "estimated_memory_mb": self._memory_usage / (1024 * 1024),
            "max_memory_mb": self.config.max_memory_mb
        }


class LazyDatasetLoader:
    """Specialized loader for dataset operations."""
    
    def __init__(self, dataset_source: Union[str, Callable], config: Optional[LazyLoadConfig] = None):
        self.dataset_source = dataset_source
        self.config = config or LazyLoadConfig()
        self.loader = LazyDataLoader(config)
        self._initialized = False
    
    def initialize(self):
        """Initialize the dataset chunks."""
        if self._initialized:
            return
        
        if callable(self.dataset_source):
            # Dynamic dataset source
            self._initialize_from_callable()
        else:
            # File-based dataset source
            self._initialize_from_file()
        
        self._initialized = True
    
    def _initialize_from_callable(self):
        """Initialize chunks from a callable data source."""
        try:
            # Get metadata about the dataset
            metadata = self.dataset_source(action="metadata")
            total_rows = metadata.get("total_rows", 0)
            
            # Create chunks based on configuration
            chunk_count = max(1, total_rows // self.config.chunk_size)
            
            for i in range(chunk_count):
                start_idx = i * self.config.chunk_size
                end_idx = min((i + 1) * self.config.chunk_size, total_rows)
                
                def make_loader(start, end):
                    return lambda: self.dataset_source(start=start, end=end)
                
                chunk_id = f"chunk_{i}"
                estimated_size = (end_idx - start_idx) * metadata.get("avg_row_size", 100)
                
                self.loader.register_chunk(chunk_id, make_loader(start_idx, end_idx), estimated_size)
                
        except Exception as e:
            logger.error(f"Failed to initialize from callable: {e}")
            raise
    
    def _initialize_from_file(self):
        """Initialize chunks from a file source."""
        import os
        
        if not os.path.exists(self.dataset_source):
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_source}")
        
        file_size = os.path.getsize(self.dataset_source)
        chunk_size_bytes = min(file_size, self.config.chunk_size * 1024)  # KB to bytes
        chunk_count = max(1, file_size // chunk_size_bytes)
        
        for i in range(chunk_count):
            start_byte = i * chunk_size_bytes
            end_byte = min((i + 1) * chunk_size_bytes, file_size)
            
            def make_file_loader(start, end, filepath):
                def load_file_chunk():
                    with open(filepath, 'rb') as f:
                        f.seek(start)
                        return f.read(end - start)
                return load_file_chunk
            
            chunk_id = f"file_chunk_{i}"
            self.loader.register_chunk(
                chunk_id, 
                make_file_loader(start_byte, end_byte, self.dataset_source),
                end_byte - start_byte
            )
    
    def get_data_iterator(self) -> Iterator[Any]:
        """Get iterator over the dataset with lazy loading."""
        if not self._initialized:
            self.initialize()
        
        return self.loader.get_chunk_iterator()
    
    def get_chunk_count(self) -> int:
        """Get the total number of chunks."""
        if not self._initialized:
            self.initialize()
        return len(self.loader.chunks)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get loading statistics."""
        stats = self.loader.get_memory_stats()
        stats["initialized"] = self._initialized
        return stats


# Utility functions for CLI integration
def create_lazy_loader_for_analysis(data_source: Any, chunk_size: int = 1000) -> LazyDatasetLoader:
    """Create a lazy loader optimized for analysis operations."""
    config = LazyLoadConfig(
        chunk_size=chunk_size,
        max_memory_mb=50,  # Conservative for CLI
        prefetch_chunks=1,
        cache_chunks=3
    )
    return LazyDatasetLoader(data_source, config)


def create_lazy_loader_for_monitoring(data_source: Any) -> LazyDatasetLoader:
    """Create a lazy loader optimized for monitoring data."""
    config = LazyLoadConfig(
        chunk_size=500,  # Smaller chunks for real-time data
        max_memory_mb=30,
        prefetch_chunks=2,
        cache_chunks=2
    )
    return LazyDatasetLoader(data_source, config)