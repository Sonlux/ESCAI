"""
Integration module for CLI performance optimization components.
Provides unified interface and coordination between all optimization systems.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
import logging

from .lazy_loader import LazyDataLoader, LazyDatasetLoader, LazyLoadConfig
from .cache_system import CLICacheManager, CacheConfig
from .memory_optimizer import CLIMemoryManager, MemoryConfig
from .parallel_processor import ParallelProcessorManager, ParallelConfig
from .progress_indicators import MultiProgressIndicator, ProgressConfig
from .resource_monitor import CLIResourceManager, ResourceConfig

logger = logging.getLogger(__name__)


@dataclass
class PerformanceConfig:
    """Unified configuration for all performance optimization components."""
    # Lazy loading configuration
    lazy_loading: LazyLoadConfig = field(default_factory=LazyLoadConfig)
    
    # Caching configuration
    caching: CacheConfig = field(default_factory=CacheConfig)
    
    # Memory optimization configuration
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    
    # Parallel processing configuration
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    
    # Progress tracking configuration
    progress: ProgressConfig = field(default_factory=ProgressConfig)
    
    # Resource monitoring configuration
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    
    # Integration settings
    enable_auto_optimization: bool = True
    optimization_interval: int = 60  # seconds
    performance_logging: bool = True


class PerformanceMetrics:
    """Tracks performance metrics across all optimization components."""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "lazy_loads": 0,
            "memory_optimizations": 0,
            "parallel_tasks": 0,
            "resource_alerts": 0
        }
        self._lock = threading.Lock()
    
    def increment(self, metric: str, value: int = 1):
        """Increment a metric counter."""
        with self._lock:
            if metric in self.metrics:
                self.metrics[metric] += value
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics with calculated rates."""
        with self._lock:
            elapsed_time = time.time() - self.start_time
            
            metrics_copy = self.metrics.copy()
            metrics_copy.update({
                "elapsed_time": elapsed_time,
                "cache_hit_rate": (
                    self.metrics["cache_hits"] / 
                    max(1, self.metrics["cache_hits"] + self.metrics["cache_misses"])
                ) * 100,
                "optimizations_per_minute": (
                    self.metrics["memory_optimizations"] / max(1, elapsed_time / 60)
                ),
                "parallel_tasks_per_second": (
                    self.metrics["parallel_tasks"] / max(1, elapsed_time)
                )
            })
            
            return metrics_copy
    
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self.start_time = time.time()
            for key in self.metrics:
                self.metrics[key] = 0


class OptimizedCLISession:
    """Manages an optimized CLI session with all performance components."""
    
    def __init__(self, session_id: str, config: Optional[PerformanceConfig] = None):
        self.session_id = session_id
        self.config = config or PerformanceConfig()
        self.metrics = PerformanceMetrics()
        
        # Initialize all performance components
        self.cache_manager = CLICacheManager(self.config.caching)
        self.memory_manager = CLIMemoryManager(self.config.memory)
        self.parallel_manager = ParallelProcessorManager(self.config.parallel)
        self.progress_manager = MultiProgressIndicator(self.config.progress)
        self.resource_manager = CLIResourceManager(self.config.resources)
        
        # Lazy loaders for different data types
        self.lazy_loaders: Dict[str, LazyDataLoader] = {}
        
        # Performance optimization state
        self._optimization_thread = None
        self._running = False
        self._callbacks: Dict[str, List[Callable]] = {
            "cache_hit": [],
            "cache_miss": [],
            "memory_optimization": [],
            "resource_alert": []
        }
    
    def start(self):
        """Start the optimized session."""
        logger.info(f"Starting optimized CLI session: {self.session_id}")
        
        # Start all monitoring systems
        self.memory_manager.start_session(self.session_id)
        self.resource_manager.start_session_monitoring(self.session_id)
        
        # Register optimization callbacks
        self._register_optimization_callbacks()
        
        # Start auto-optimization if enabled
        if self.config.enable_auto_optimization:
            self._start_auto_optimization()
        
        self._running = True
    
    def stop(self) -> Dict[str, Any]:
        """Stop the session and return performance summary."""
        logger.info(f"Stopping optimized CLI session: {self.session_id}")
        
        self._running = False
        
        # Stop auto-optimization
        if self._optimization_thread and self._optimization_thread.is_alive():
            self._optimization_thread.join(timeout=5.0)
        
        # Get final summaries
        memory_summary = self.memory_manager.end_session(self.session_id)
        resource_summary = self.resource_manager.end_session_monitoring(self.session_id)
        cache_stats = self.cache_manager.get_stats()
        metrics = self.metrics.get_metrics()
        
        # Compile comprehensive summary
        summary = {
            "session_id": self.session_id,
            "duration": metrics["elapsed_time"],
            "memory_summary": memory_summary,
            "resource_summary": resource_summary,
            "cache_stats": cache_stats,
            "performance_metrics": metrics,
            "lazy_loaders": {
                name: loader.get_memory_stats() 
                for name, loader in self.lazy_loaders.items()
            }
        }
        
        if self.config.performance_logging:
            logger.info(f"Session performance summary: {summary}")
        
        return summary
    
    def create_lazy_loader(self, name: str, data_source: Any, 
                          config: Optional[LazyLoadConfig] = None) -> LazyDatasetLoader:
        """Create a lazy loader for a data source."""
        loader_config = config or self.config.lazy_loading
        loader = LazyDatasetLoader(data_source, loader_config)
        
        # Wrap loader to track metrics
        original_get_data_iterator = loader.get_data_iterator
        
        def tracked_get_data_iterator():
            self.metrics.increment("lazy_loads")
            return original_get_data_iterator()
        
        loader.get_data_iterator = tracked_get_data_iterator
        self.lazy_loaders[name] = loader
        
        return loader
    
    def cached_operation(self, key: str, operation: Callable, ttl: Optional[int] = None) -> Any:
        """Perform a cached operation."""
        # Try cache first
        cached_result = self.cache_manager.get(key)
        if cached_result is not None:
            self.metrics.increment("cache_hits")
            self._trigger_callbacks("cache_hit", key, cached_result)
            return cached_result
        
        # Cache miss - execute operation
        self.metrics.increment("cache_misses")
        result = operation()
        
        # Cache the result
        self.cache_manager.put(key, result, ttl)
        self._trigger_callbacks("cache_miss", key, result)
        
        return result
    
    def parallel_process(self, data: List[Any], func: Callable, 
                        processor_type: Optional[str] = None, **kwargs) -> List[Any]:
        """Process data in parallel with progress tracking."""
        # Determine optimal processor type
        if processor_type is None:
            processor_type = self.parallel_manager.get_optimal_processor_type("analysis")
        
        # Add progress tracking
        progress_name = f"parallel_process_{int(time.time())}"
        progress_indicator = self.progress_manager.add_progress(
            progress_name, len(data), f"Processing {len(data)} items"
        )
        
        # Define progress callback
        def progress_callback(completed_count):
            self.progress_manager.update_progress(progress_name, completed_count)
        
        # Process with tracking
        self.metrics.increment("parallel_tasks", len(data))
        results = self.parallel_manager.process_analysis_data(
            data, func, processor_type, **kwargs
        )
        
        # Update final progress
        successful_count = sum(1 for r in results if r.success)
        self.progress_manager.update_progress(progress_name, successful_count)
        
        return results
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Force memory optimization."""
        result = self.memory_manager.optimize_memory()
        self.metrics.increment("memory_optimizations")
        self._trigger_callbacks("memory_optimization", result)
        return result
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            "session_id": self.session_id,
            "metrics": self.metrics.get_metrics(),
            "cache_stats": self.cache_manager.get_stats(),
            "memory_report": self.memory_manager.get_memory_report(),
            "resource_report": self.resource_manager.get_resource_manager().get_monitoring_report(),
            "parallel_stats": {
                "optimal_processor": self.parallel_manager.get_optimal_processor_type("analysis")
            },
            "lazy_loader_stats": {
                name: loader.get_stats() if hasattr(loader, 'get_stats') else {}
                for name, loader in self.lazy_loaders.items()
            }
        }
    
    def register_callback(self, event_type: str, callback: Callable):
        """Register a callback for performance events."""
        if event_type in self._callbacks:
            self._callbacks[event_type].append(callback)
    
    def _register_optimization_callbacks(self):
        """Register callbacks with optimization components."""
        # Resource monitoring alerts
        def resource_alert_callback(level, resource, value, usage):
            self.metrics.increment("resource_alerts")
            self._trigger_callbacks("resource_alert", level, resource, value)
        
        self.resource_manager.monitor.register_alert_callback(resource_alert_callback)
        
        # Memory optimization callbacks
        def memory_optimization_callback():
            return "Session-triggered memory optimization"
        
        self.memory_manager.optimizer.register_optimization_callback(memory_optimization_callback)
    
    def _start_auto_optimization(self):
        """Start automatic optimization thread."""
        def optimization_worker():
            while self._running:
                try:
                    time.sleep(self.config.optimization_interval)
                    
                    if not self._running:
                        break
                    
                    # Check if optimization is needed
                    metrics = self.metrics.get_metrics()
                    
                    # Auto-optimize based on metrics
                    if metrics["cache_hit_rate"] < 50:  # Low cache hit rate
                        logger.info("Auto-optimization: Low cache hit rate detected")
                    
                    if metrics["memory_optimizations"] == 0 and metrics["elapsed_time"] > 300:
                        # No memory optimizations in 5+ minutes
                        self.optimize_memory()
                        logger.info("Auto-optimization: Performed memory optimization")
                    
                except Exception as e:
                    logger.error(f"Auto-optimization error: {e}")
        
        self._optimization_thread = threading.Thread(target=optimization_worker, daemon=True)
        self._optimization_thread.start()
    
    def _trigger_callbacks(self, event_type: str, *args):
        """Trigger callbacks for an event type."""
        if event_type in self._callbacks:
            for callback in self._callbacks[event_type]:
                try:
                    callback(*args)
                except Exception as e:
                    logger.error(f"Callback error for {event_type}: {e}")


class PerformanceOptimizer:
    """High-level performance optimizer for CLI operations."""
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        self.active_sessions: Dict[str, OptimizedCLISession] = {}
        self._global_metrics = PerformanceMetrics()
    
    def create_session(self, session_id: str) -> OptimizedCLISession:
        """Create a new optimized session."""
        if session_id in self.active_sessions:
            raise ValueError(f"Session {session_id} already exists")
        
        session = OptimizedCLISession(session_id, self.config)
        self.active_sessions[session_id] = session
        
        return session
    
    def get_session(self, session_id: str) -> Optional[OptimizedCLISession]:
        """Get an existing session."""
        return self.active_sessions.get(session_id)
    
    def end_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """End a session and return its summary."""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        summary = session.stop()
        
        del self.active_sessions[session_id]
        
        return summary
    
    def get_global_performance_report(self) -> Dict[str, Any]:
        """Get global performance report across all sessions."""
        return {
            "active_sessions": len(self.active_sessions),
            "session_summaries": {
                session_id: session.get_performance_report()
                for session_id, session in self.active_sessions.items()
            },
            "global_metrics": self._global_metrics.get_metrics()
        }
    
    @contextmanager
    def optimized_session(self, session_id: str):
        """Context manager for optimized sessions."""
        session = self.create_session(session_id)
        session.start()
        
        try:
            yield session
        finally:
            self.end_session(session_id)


# Global performance optimizer instance
_performance_optimizer: Optional[PerformanceOptimizer] = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get the global performance optimizer."""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer


@contextmanager
def optimized_cli_session(session_id: str, config: Optional[PerformanceConfig] = None):
    """Context manager for creating optimized CLI sessions."""
    optimizer = get_performance_optimizer()
    if config:
        optimizer.config = config
    
    with optimizer.optimized_session(session_id) as session:
        yield session


# Convenience functions for common operations
def create_optimized_analysis(session_id: str, data_source: Any, 
                            analysis_func: Callable, **kwargs) -> Any:
    """Create an optimized analysis operation."""
    with optimized_cli_session(session_id) as session:
        # Create lazy loader for data
        loader = session.create_lazy_loader("analysis_data", data_source)
        
        # Get data iterator
        data_iterator = loader.get_data_iterator()
        data_chunks = list(data_iterator)
        
        # Process in parallel with caching
        cache_key = f"analysis_{session_id}_{hash(str(kwargs))}"
        
        def cached_analysis():
            return session.parallel_process(data_chunks, analysis_func, **kwargs)
        
        results = session.cached_operation(cache_key, cached_analysis, ttl=3600)
        
        return results


def optimize_cli_performance(func: Callable):
    """Decorator for optimizing CLI function performance."""
    def wrapper(*args, **kwargs):
        session_id = f"optimized_{func.__name__}_{int(time.time())}"
        
        with optimized_cli_session(session_id) as session:
            # Track the operation
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Log performance
                duration = time.time() - start_time
                logger.info(f"Optimized function {func.__name__} completed in {duration:.2f}s")
                
                return result
                
            except Exception as e:
                logger.error(f"Optimized function {func.__name__} failed: {e}")
                raise
    
    return wrapper