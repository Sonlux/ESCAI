"""
Memory optimization system for long-running CLI sessions.
Implements memory monitoring, garbage collection, and resource management.
"""

import gc
import psutil
import threading
import time
import weakref
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from collections import deque
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for memory optimization."""
    max_memory_mb: int = 200  # Maximum memory usage
    warning_threshold: float = 0.8  # Warning at 80% of max
    critical_threshold: float = 0.95  # Critical at 95% of max
    gc_interval: int = 30  # Garbage collection interval in seconds
    monitoring_interval: int = 10  # Memory monitoring interval
    enable_aggressive_gc: bool = True  # Enable aggressive garbage collection
    max_session_objects: int = 1000  # Maximum objects to keep in session


class MemoryStats:
    """Memory usage statistics."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = self.get_current_memory()
        self.peak_memory = self.baseline_memory
        self.gc_count = 0
        self.optimization_count = 0
    
    def get_current_memory(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / (1024 * 1024)
    
    def get_memory_percent(self) -> float:
        """Get memory usage as percentage of system memory."""
        return self.process.memory_percent()
    
    def update_peak(self):
        """Update peak memory usage."""
        current = self.get_current_memory()
        if current > self.peak_memory:
            self.peak_memory = current
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        current_memory = self.get_current_memory()
        return {
            "current_mb": current_memory,
            "baseline_mb": self.baseline_memory,
            "peak_mb": self.peak_memory,
            "growth_mb": current_memory - self.baseline_memory,
            "system_percent": self.get_memory_percent(),
            "gc_collections": self.gc_count,
            "optimizations": self.optimization_count
        }


class SessionObjectTracker:
    """Tracks objects created during CLI session for cleanup."""
    
    def __init__(self, max_objects: int = 1000):
        self.max_objects = max_objects
        self.tracked_objects: deque = deque(maxlen=max_objects)
        self.object_refs: weakref.WeakSet = weakref.WeakSet()
        self._lock = threading.Lock()
    
    def track_object(self, obj: Any, category: str = "general"):
        """Track an object for potential cleanup."""
        with self._lock:
            try:
                ref = weakref.ref(obj)
                self.object_refs.add(obj)
                self.tracked_objects.append({
                    "ref": ref,
                    "category": category,
                    "created_at": time.time(),
                    "type": type(obj).__name__
                })
            except TypeError:
                # Object doesn't support weak references
                pass
    
    def cleanup_category(self, category: str) -> int:
        """Clean up objects in a specific category."""
        cleaned = 0
        with self._lock:
            # Create new deque without the specified category
            new_objects: deque[Dict[str, Any]] = deque(maxlen=self.max_objects)
            
            for obj_info in self.tracked_objects:
                if obj_info["category"] == category:
                    # Try to delete the object
                    ref = obj_info["ref"]
                    if ref() is not None:
                        try:
                            del ref
                            cleaned += 1
                        except:
                            pass
                else:
                    new_objects.append(obj_info)
            
            self.tracked_objects = new_objects
        
        return cleaned
    
    def cleanup_old_objects(self, max_age_seconds: int = 3600) -> int:
        """Clean up objects older than specified age."""
        current_time = time.time()
        cleaned = 0
        
        with self._lock:
            new_objects: deque[Dict[str, Any]] = deque(maxlen=self.max_objects)
            
            for obj_info in self.tracked_objects:
                age = current_time - obj_info["created_at"]
                if age > max_age_seconds:
                    ref = obj_info["ref"]
                    if ref() is not None:
                        try:
                            del ref
                            cleaned += 1
                        except:
                            pass
                else:
                    new_objects.append(obj_info)
            
            self.tracked_objects = new_objects
        
        return cleaned
    
    def get_object_stats(self) -> Dict[str, Any]:
        """Get statistics about tracked objects."""
        with self._lock:
            categories = {}
            alive_count = 0
            
            for obj_info in self.tracked_objects:
                category = obj_info["category"]
                if category not in categories:
                    categories[category] = 0
                categories[category] += 1
                
                if obj_info["ref"]() is not None:
                    alive_count += 1
            
            return {
                "total_tracked": len(self.tracked_objects),
                "alive_objects": alive_count,
                "categories": categories
            }


class MemoryOptimizer:
    """Main memory optimization system."""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.stats = MemoryStats()
        self.object_tracker = SessionObjectTracker(self.config.max_session_objects)
        self.optimization_callbacks: List[Callable] = []
        
        self._monitoring_thread = None
        self._gc_thread = None
        self._running = False
        self._lock = threading.Lock()
    
    def start_monitoring(self):
        """Start memory monitoring and optimization."""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            
            # Start monitoring thread
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_worker, daemon=True
            )
            self._monitoring_thread.start()
            
            # Start garbage collection thread
            self._gc_thread = threading.Thread(
                target=self._gc_worker, daemon=True
            )
            self._gc_thread.start()
            
            logger.info("Memory optimization started")
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        with self._lock:
            self._running = False
            logger.info("Memory optimization stopped")
    
    def register_optimization_callback(self, callback: Callable):
        """Register a callback for memory optimization events."""
        self.optimization_callbacks.append(callback)
    
    def track_object(self, obj: Any, category: str = "general"):
        """Track an object for memory management."""
        self.object_tracker.track_object(obj, category)
    
    def force_optimization(self) -> Dict[str, Any]:
        """Force immediate memory optimization."""
        logger.info("Forcing memory optimization")
        
        results = {
            "gc_collected": 0,
            "objects_cleaned": 0,
            "memory_before": self.stats.get_current_memory(),
            "memory_after": 0
        }
        
        # Run garbage collection
        if self.config.enable_aggressive_gc:
            results["gc_collected"] = self._run_garbage_collection()
        
        # Clean up old objects
        results["objects_cleaned"] = self.object_tracker.cleanup_old_objects(1800)  # 30 minutes
        
        # Run optimization callbacks
        for callback in self.optimization_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Optimization callback error: {e}")
        
        results["memory_after"] = self.stats.get_current_memory()
        self.stats.optimization_count += 1
        
        logger.info(f"Memory optimization completed: {results}")
        return results
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory status."""
        current_memory = self.stats.get_current_memory()
        max_memory = self.config.max_memory_mb
        
        status = {
            "current_mb": current_memory,
            "max_mb": max_memory,
            "usage_percent": (current_memory / max_memory) * 100,
            "status": "normal"
        }
        
        usage_ratio = current_memory / max_memory
        
        if usage_ratio >= self.config.critical_threshold:
            status["status"] = "critical"
        elif usage_ratio >= self.config.warning_threshold:
            status["status"] = "warning"
        
        return status
    
    def _monitoring_worker(self):
        """Background worker for memory monitoring."""
        while self._running:
            try:
                self.stats.update_peak()
                current_memory = self.stats.get_current_memory()
                max_memory = self.config.max_memory_mb
                usage_ratio = current_memory / max_memory
                
                # Check if optimization is needed
                if usage_ratio >= self.config.critical_threshold:
                    logger.warning(f"Critical memory usage: {current_memory:.1f}MB ({usage_ratio*100:.1f}%)")
                    self.force_optimization()
                elif usage_ratio >= self.config.warning_threshold:
                    logger.warning(f"High memory usage: {current_memory:.1f}MB ({usage_ratio*100:.1f}%)")
                
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(self.config.monitoring_interval)
    
    def _gc_worker(self):
        """Background worker for garbage collection."""
        while self._running:
            try:
                time.sleep(self.config.gc_interval)
                
                if self.config.enable_aggressive_gc:
                    collected = self._run_garbage_collection()
                    if collected > 0:
                        logger.debug(f"Garbage collection freed {collected} objects")
                
            except Exception as e:
                logger.error(f"Garbage collection error: {e}")
    
    def _run_garbage_collection(self) -> int:
        """Run garbage collection and return number of collected objects."""
        # Force garbage collection
        collected = 0
        for generation in range(3):
            collected += gc.collect(generation)
        
        self.stats.gc_count += 1
        return collected
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory and optimization statistics."""
        return {
            "memory": self.stats.get_stats(),
            "objects": self.object_tracker.get_object_stats(),
            "status": self.get_memory_status(),
            "config": {
                "max_memory_mb": self.config.max_memory_mb,
                "warning_threshold": self.config.warning_threshold,
                "critical_threshold": self.config.critical_threshold
            }
        }


class CLIMemoryManager:
    """High-level memory manager for CLI operations."""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.optimizer = MemoryOptimizer(config)
        self.session_data: Dict[str, Any] = {}
        self._cleanup_handlers: Dict[str, Callable] = {}
    
    def start_session(self, session_id: str):
        """Start a new memory-managed session."""
        self.optimizer.start_monitoring()
        self.session_data[session_id] = {
            "start_time": time.time(),
            "objects": []
        }
        logger.info(f"Started memory-managed session: {session_id}")
    
    def end_session(self, session_id: str):
        """End a session and clean up its resources."""
        if session_id in self.session_data:
            # Clean up session objects
            self.optimizer.object_tracker.cleanup_category(f"session_{session_id}")
            
            # Run cleanup handlers
            if session_id in self._cleanup_handlers:
                try:
                    self._cleanup_handlers[session_id]()
                except Exception as e:
                    logger.error(f"Session cleanup error: {e}")
            
            del self.session_data[session_id]
            logger.info(f"Ended memory-managed session: {session_id}")
        
        # Stop monitoring if no active sessions
        if not self.session_data:
            self.optimizer.stop_monitoring()
    
    def register_session_cleanup(self, session_id: str, cleanup_func: Callable):
        """Register a cleanup function for a session."""
        self._cleanup_handlers[session_id] = cleanup_func
    
    def track_session_object(self, session_id: str, obj: Any, name: str = ""):
        """Track an object for a specific session."""
        category = f"session_{session_id}"
        self.optimizer.track_object(obj, category)
        
        if session_id in self.session_data:
            self.session_data[session_id]["objects"].append({
                "name": name,
                "type": type(obj).__name__,
                "created_at": time.time()
            })
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Force memory optimization."""
        return self.optimizer.force_optimization()
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory report."""
        report = self.optimizer.get_comprehensive_stats()
        report["sessions"] = {
            session_id: {
                "duration": time.time() - data["start_time"],
                "object_count": len(data["objects"])
            }
            for session_id, data in self.session_data.items()
        }
        return report


# Global memory manager instance
_memory_manager: Optional[CLIMemoryManager] = None


def get_memory_manager() -> CLIMemoryManager:
    """Get the global CLI memory manager."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = CLIMemoryManager()
    return _memory_manager


def memory_optimized_session(session_id: str):
    """Context manager for memory-optimized sessions."""
    class MemoryOptimizedSession:
        def __enter__(self):
            get_memory_manager().start_session(session_id)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            get_memory_manager().end_session(session_id)
    
    return MemoryOptimizedSession()


def track_cli_object(obj: Any, session_id: str = "default", name: str = ""):
    """Track an object for memory management."""
    get_memory_manager().track_session_object(session_id, obj, name)