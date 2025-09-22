"""
CLI startup optimization for fast launch times.

This module provides various optimizations to reduce CLI startup time
and improve user experience.
"""

import os
import sys
import time
import threading
import importlib
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from functools import lru_cache
import pickle
import hashlib

from .console import get_console


class LazyImporter:
    """
    Lazy importer that defers module imports until they're actually needed.
    
    This helps reduce startup time by avoiding expensive imports during CLI initialization.
    """
    
    def __init__(self, module_name: str, attribute: Optional[str] = None):
        """
        Initialize lazy importer.
        
        Args:
            module_name: Name of module to import
            attribute: Optional attribute to get from module
        """
        self.module_name = module_name
        self.attribute = attribute
        self._module = None
        self._lock = threading.Lock()
    
    def __call__(self, *args, **kwargs):
        """Import and call the module/attribute."""
        obj = self._get_object()
        if callable(obj):
            return obj(*args, **kwargs)
        return obj
    
    def __getattr__(self, name):
        """Get attribute from the imported module."""
        obj = self._get_object()
        return getattr(obj, name)
    
    def _get_object(self):
        """Get the actual module or attribute."""
        if self._module is None:
            with self._lock:
                if self._module is None:
                    module = importlib.import_module(self.module_name)
                    if self.attribute:
                        self._module = getattr(module, self.attribute)
                    else:
                        self._module = module
        return self._module


class StartupCache:
    """
    Cache for expensive startup operations.
    
    Caches results of expensive operations like framework detection,
    configuration validation, etc.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize startup cache.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir or Path.home() / ".escai" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, Any] = {}
        self._cache_ttl = 3600  # 1 hour default TTL
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        # Check memory cache first
        if key in self._memory_cache:
            return self._memory_cache[key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{self._hash_key(key)}.cache"
        if cache_file.exists():
            try:
                # Check if cache is still valid
                if time.time() - cache_file.stat().st_mtime < self._cache_ttl:
                    with open(cache_file, 'rb') as f:
                        value = pickle.load(f)
                        self._memory_cache[key] = value
                        return value
                else:
                    # Cache expired, remove it
                    cache_file.unlink()
            except Exception:
                # Cache corrupted, remove it
                try:
                    cache_file.unlink()
                except:
                    pass
        
        return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Store in memory cache
        self._memory_cache[key] = value
        
        # Store in disk cache
        cache_file = self.cache_dir / f"{self._hash_key(key)}.cache"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception:
            # Ignore cache write errors
            pass
    
    def invalidate(self, key: str) -> None:
        """
        Invalidate cache entry.
        
        Args:
            key: Cache key to invalidate
        """
        # Remove from memory cache
        self._memory_cache.pop(key, None)
        
        # Remove from disk cache
        cache_file = self.cache_dir / f"{self._hash_key(key)}.cache"
        try:
            cache_file.unlink()
        except FileNotFoundError:
            pass
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._memory_cache.clear()
        
        # Remove all cache files
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                cache_file.unlink()
            except:
                pass
    
    def _hash_key(self, key: str) -> str:
        """Generate hash for cache key."""
        return hashlib.md5(key.encode()).hexdigest()


class StartupProfiler:
    """
    Profiler for measuring startup performance.
    
    Helps identify bottlenecks in CLI startup process.
    """
    
    def __init__(self, enabled: bool = False):
        """
        Initialize startup profiler.
        
        Args:
            enabled: Whether profiling is enabled
        """
        self.enabled = enabled
        self._start_time = time.time()
        self._checkpoints: List[tuple] = []
        self._console = get_console()
    
    def checkpoint(self, name: str) -> None:
        """
        Record a checkpoint.
        
        Args:
            name: Checkpoint name
        """
        if not self.enabled:
            return
        
        current_time = time.time()
        elapsed = current_time - self._start_time
        self._checkpoints.append((name, elapsed))
    
    def report(self) -> None:
        """Report startup performance."""
        if not self.enabled or not self._checkpoints:
            return
        
        total_time = time.time() - self._start_time
        
        self._console.print(f"\n[dim]Startup Performance Report (Total: {total_time:.3f}s)[/dim]")
        
        prev_time = 0
        for name, elapsed in self._checkpoints:
            delta = elapsed - prev_time
            self._console.print(f"[dim]  {name}: {delta:.3f}s (cumulative: {elapsed:.3f}s)[/dim]")
            prev_time = elapsed


class StartupOptimizer:
    """
    Main startup optimizer that coordinates various optimization strategies.
    """
    
    def __init__(self):
        """Initialize startup optimizer."""
        self.cache = StartupCache()
        self.profiler = StartupProfiler(enabled=os.getenv('ESCAI_PROFILE_STARTUP') == '1')
        self._lazy_imports: Dict[str, LazyImporter] = {}
        self._preload_tasks: List[Callable] = []
        self._background_thread: Optional[threading.Thread] = None
        
        # Start profiling
        self.profiler.checkpoint("optimizer_init")
    
    def register_lazy_import(self, name: str, module_name: str, attribute: Optional[str] = None) -> LazyImporter:
        """
        Register a lazy import.
        
        Args:
            name: Name to register under
            module_name: Module to import
            attribute: Optional attribute to get from module
            
        Returns:
            LazyImporter instance
        """
        lazy_importer = LazyImporter(module_name, attribute)
        self._lazy_imports[name] = lazy_importer
        return lazy_importer
    
    def add_preload_task(self, task: Callable) -> None:
        """
        Add a task to be preloaded in the background.
        
        Args:
            task: Task function to execute
        """
        self._preload_tasks.append(task)
    
    def start_background_preload(self) -> None:
        """Start background preloading of expensive operations."""
        if self._background_thread is not None:
            return
        
        def preload_worker():
            """Background worker for preloading tasks."""
            for task in self._preload_tasks:
                try:
                    task()
                except Exception:
                    # Ignore preload errors - they're optimizations
                    pass
        
        self._background_thread = threading.Thread(target=preload_worker, daemon=True)
        self._background_thread.start()
        
        self.profiler.checkpoint("background_preload_started")
    
    @lru_cache(maxsize=128)
    def get_framework_availability(self, framework: str) -> bool:
        """
        Get framework availability with caching.
        
        Args:
            framework: Framework name
            
        Returns:
            True if framework is available
        """
        cache_key = f"framework_availability_{framework}"
        cached = self.cache.get(cache_key)
        
        if cached is not None:
            return cached
        
        # Check framework availability
        available = False
        try:
            if framework == 'langchain':
                import langchain
                available = True
            elif framework == 'autogen':
                import autogen
                available = True
            elif framework == 'crewai':
                import crewai
                available = True
            elif framework == 'openai':
                import openai
                available = True
        except ImportError:
            available = False
        
        # Cache result
        self.cache.set(cache_key, available)
        return available
    
    def optimize_imports(self) -> None:
        """Optimize imports by deferring expensive ones."""
        # Replace expensive imports with lazy imports
        self.register_lazy_import('pandas', 'pandas')
        self.register_lazy_import('numpy', 'numpy')
        self.register_lazy_import('matplotlib', 'matplotlib.pyplot')
        self.register_lazy_import('plotly', 'plotly.graph_objects')
        
        self.profiler.checkpoint("imports_optimized")
    
    def preload_configuration(self) -> None:
        """Preload configuration in background."""
        def load_config():
            """Load configuration."""
            try:
                from ..config.config_manager import get_config_manager
                config_manager = get_config_manager()
                config_manager.get_active_profile()
            except Exception:
                pass
        
        self.add_preload_task(load_config)
    
    def preload_database_connection(self) -> None:
        """Preload database connection in background."""
        def load_db():
            """Load database connection."""
            try:
                from ...storage.database import get_database_manager
                db_manager = get_database_manager()
                # Just initialize, don't actually connect
                db_manager.initialize()
            except Exception:
                pass
        
        self.add_preload_task(load_db)
    
    def optimize_console_initialization(self) -> None:
        """Optimize console initialization."""
        # Pre-initialize console components
        try:
            console = get_console()
            # Trigger lazy initialization
            console.size
        except Exception:
            pass
        
        self.profiler.checkpoint("console_optimized")
    
    def cleanup_old_cache(self) -> None:
        """Clean up old cache files."""
        def cleanup():
            """Cleanup worker."""
            try:
                cache_dir = self.cache.cache_dir
                current_time = time.time()
                
                for cache_file in cache_dir.glob("*.cache"):
                    try:
                        # Remove files older than 24 hours
                        if current_time - cache_file.stat().st_mtime > 86400:
                            cache_file.unlink()
                    except Exception:
                        pass
            except Exception:
                pass
        
        self.add_preload_task(cleanup)
    
    def finalize_startup(self) -> None:
        """Finalize startup optimization."""
        self.profiler.checkpoint("startup_complete")
        self.profiler.report()
    
    def get_startup_stats(self) -> Dict[str, Any]:
        """
        Get startup statistics.
        
        Returns:
            Dictionary with startup statistics
        """
        return {
            'cache_hits': len(self.cache._memory_cache),
            'lazy_imports': len(self._lazy_imports),
            'preload_tasks': len(self._preload_tasks),
            'background_thread_active': self._background_thread is not None and self._background_thread.is_alive(),
            'checkpoints': len(self.profiler._checkpoints)
        }


# Global optimizer instance
_optimizer: Optional[StartupOptimizer] = None


def get_startup_optimizer() -> StartupOptimizer:
    """Get the global startup optimizer instance."""
    global _optimizer
    if _optimizer is None:
        _optimizer = StartupOptimizer()
    return _optimizer


def optimize_cli_startup() -> None:
    """
    Optimize CLI startup performance.
    
    This function should be called early in the CLI initialization process.
    """
    optimizer = get_startup_optimizer()
    
    # Apply optimizations
    optimizer.optimize_imports()
    optimizer.optimize_console_initialization()
    
    # Setup background preloading
    optimizer.preload_configuration()
    optimizer.preload_database_connection()
    optimizer.cleanup_old_cache()
    
    # Start background tasks
    optimizer.start_background_preload()
    
    # Mark optimization complete
    optimizer.profiler.checkpoint("optimization_complete")


def finalize_cli_startup() -> None:
    """
    Finalize CLI startup optimization.
    
    This function should be called after CLI initialization is complete.
    """
    optimizer = get_startup_optimizer()
    optimizer.finalize_startup()