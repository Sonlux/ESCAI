"""
Parallel processing system for CLI analysis operations.
Implements efficient parallel execution with progress tracking and resource management.
"""

import asyncio
import concurrent.futures
import multiprocessing
import threading
import time
from typing import Any, Dict, List, Optional, Callable, Union, Iterator, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import queue
import logging
from functools import partial

logger = logging.getLogger(__name__)


@dataclass
class ParallelConfig:
    """Configuration for parallel processing."""
    max_workers: int = None  # None = auto-detect
    chunk_size: int = 100  # Items per chunk
    timeout_seconds: int = 300  # 5 minutes default timeout
    enable_progress_tracking: bool = True
    use_process_pool: bool = False  # Use threads by default for CLI
    memory_limit_mb: int = 100  # Memory limit per worker


class ProcessingResult:
    """Result of a parallel processing operation."""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.success = False
        self.result: Any = None
        self.error: Optional[Exception] = None
        self.worker_id: Optional[str] = None
    
    def complete(self, result: Any, worker_id: str = ""):
        """Mark the result as completed successfully."""
        self.end_time = time.time()
        self.success = True
        self.result = result
        self.worker_id = worker_id
    
    def fail(self, error: Exception, worker_id: str = ""):
        """Mark the result as failed."""
        self.end_time = time.time()
        self.success = False
        self.error = error
        self.worker_id = worker_id
    
    def get_duration(self) -> float:
        """Get the processing duration in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "task_id": self.task_id,
            "success": self.success,
            "duration": self.get_duration(),
            "worker_id": self.worker_id,
            "error": str(self.error) if self.error else None
        }


class ProgressTracker:
    """Tracks progress of parallel operations."""
    
    def __init__(self, total_tasks: int):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.start_time = time.time()
        self._lock = threading.Lock()
        self.callbacks: List[Callable] = []
    
    def add_callback(self, callback: Callable):
        """Add a progress callback function."""
        self.callbacks.append(callback)
    
    def update(self, completed: int = 1, failed: int = 0):
        """Update progress counters."""
        with self._lock:
            self.completed_tasks += completed
            self.failed_tasks += failed
            
            # Call progress callbacks
            for callback in self.callbacks:
                try:
                    callback(self.get_progress_info())
                except Exception as e:
                    logger.error(f"Progress callback error: {e}")
    
    def get_progress_info(self) -> Dict[str, Any]:
        """Get current progress information."""
        with self._lock:
            processed = self.completed_tasks + self.failed_tasks
            progress_percent = (processed / self.total_tasks * 100) if self.total_tasks > 0 else 0
            
            elapsed_time = time.time() - self.start_time
            if processed > 0:
                avg_time_per_task = elapsed_time / processed
                estimated_remaining = avg_time_per_task * (self.total_tasks - processed)
            else:
                estimated_remaining = 0
            
            return {
                "total_tasks": self.total_tasks,
                "completed": self.completed_tasks,
                "failed": self.failed_tasks,
                "processed": processed,
                "progress_percent": progress_percent,
                "elapsed_seconds": elapsed_time,
                "estimated_remaining_seconds": estimated_remaining,
                "tasks_per_second": processed / elapsed_time if elapsed_time > 0 else 0
            }
    
    def is_complete(self) -> bool:
        """Check if all tasks are complete."""
        with self._lock:
            return (self.completed_tasks + self.failed_tasks) >= self.total_tasks


class ParallelProcessor(ABC):
    """Abstract base class for parallel processors."""
    
    @abstractmethod
    def process_batch(self, items: List[Any], func: Callable, **kwargs) -> List[ProcessingResult]:
        pass
    
    @abstractmethod
    def process_stream(self, items: Iterator[Any], func: Callable, **kwargs) -> Iterator[ProcessingResult]:
        pass


class ThreadPoolProcessor(ParallelProcessor):
    """Thread-based parallel processor for I/O bound tasks."""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.max_workers = config.max_workers or min(32, (multiprocessing.cpu_count() or 1) + 4)
    
    def process_batch(self, items: List[Any], func: Callable, **kwargs) -> List[ProcessingResult]:
        """Process a batch of items in parallel using threads."""
        results = []
        progress_tracker = ProgressTracker(len(items)) if self.config.enable_progress_tracking else None
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_item = {}
            for i, item in enumerate(items):
                task_id = f"task_{i}"
                future = executor.submit(self._safe_execute, func, item, task_id, **kwargs)
                future_to_item[future] = (item, task_id)
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_item, timeout=self.config.timeout_seconds):
                item, task_id = future_to_item[future]
                result = ProcessingResult(task_id)
                
                try:
                    func_result = future.result()
                    result.complete(func_result, f"thread_{threading.current_thread().ident}")
                    if progress_tracker:
                        progress_tracker.update(completed=1)
                except Exception as e:
                    result.fail(e, f"thread_{threading.current_thread().ident}")
                    if progress_tracker:
                        progress_tracker.update(failed=1)
                    logger.error(f"Task {task_id} failed: {e}")
                
                results.append(result)
        
        return results
    
    def process_stream(self, items: Iterator[Any], func: Callable, **kwargs) -> Iterator[ProcessingResult]:
        """Process a stream of items in parallel using threads."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            task_counter = 0
            
            # Submit initial batch
            for _ in range(self.max_workers * 2):  # Buffer 2x workers
                try:
                    item = next(items)
                    task_id = f"stream_task_{task_counter}"
                    future = executor.submit(self._safe_execute, func, item, task_id, **kwargs)
                    futures[future] = (item, task_id)
                    task_counter += 1
                except StopIteration:
                    break
            
            # Process and yield results
            while futures:
                # Wait for at least one to complete
                done_futures = concurrent.futures.as_completed(futures, timeout=1.0)
                
                for future in done_futures:
                    item, task_id = futures.pop(future)
                    result = ProcessingResult(task_id)
                    
                    try:
                        func_result = future.result()
                        result.complete(func_result, f"thread_{threading.current_thread().ident}")
                    except Exception as e:
                        result.fail(e, f"thread_{threading.current_thread().ident}")
                        logger.error(f"Stream task {task_id} failed: {e}")
                    
                    yield result
                    
                    # Submit next item if available
                    try:
                        next_item = next(items)
                        next_task_id = f"stream_task_{task_counter}"
                        next_future = executor.submit(self._safe_execute, func, next_item, next_task_id, **kwargs)
                        futures[next_future] = (next_item, next_task_id)
                        task_counter += 1
                    except StopIteration:
                        pass
    
    def _safe_execute(self, func: Callable, item: Any, task_id: str, **kwargs) -> Any:
        """Safely execute a function with error handling."""
        try:
            return func(item, **kwargs)
        except Exception as e:
            logger.error(f"Function execution failed for {task_id}: {e}")
            raise


class ProcessPoolProcessor(ParallelProcessor):
    """Process-based parallel processor for CPU-bound tasks."""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.max_workers = config.max_workers or multiprocessing.cpu_count()
    
    def process_batch(self, items: List[Any], func: Callable, **kwargs) -> List[ProcessingResult]:
        """Process a batch of items in parallel using processes."""
        results = []
        progress_tracker = ProgressTracker(len(items)) if self.config.enable_progress_tracking else None
        
        # Prepare function with kwargs
        worker_func = partial(self._process_worker, func, **kwargs)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_item = {}
            for i, item in enumerate(items):
                task_id = f"task_{i}"
                future = executor.submit(worker_func, item, task_id)
                future_to_item[future] = (item, task_id)
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_item, timeout=self.config.timeout_seconds):
                item, task_id = future_to_item[future]
                result = ProcessingResult(task_id)
                
                try:
                    func_result, worker_id = future.result()
                    result.complete(func_result, worker_id)
                    if progress_tracker:
                        progress_tracker.update(completed=1)
                except Exception as e:
                    result.fail(e, f"process_{multiprocessing.current_process().pid}")
                    if progress_tracker:
                        progress_tracker.update(failed=1)
                    logger.error(f"Task {task_id} failed: {e}")
                
                results.append(result)
        
        return results
    
    def process_stream(self, items: Iterator[Any], func: Callable, **kwargs) -> Iterator[ProcessingResult]:
        """Process a stream of items in parallel using processes."""
        # Convert iterator to chunks for process pool
        chunk_size = self.config.chunk_size
        current_chunk = []
        
        for item in items:
            current_chunk.append(item)
            
            if len(current_chunk) >= chunk_size:
                # Process current chunk
                chunk_results = self.process_batch(current_chunk, func, **kwargs)
                for result in chunk_results:
                    yield result
                current_chunk = []
        
        # Process remaining items
        if current_chunk:
            chunk_results = self.process_batch(current_chunk, func, **kwargs)
            for result in chunk_results:
                yield result
    
    @staticmethod
    def _process_worker(func: Callable, item: Any, task_id: str, **kwargs) -> Tuple[Any, str]:
        """Worker function for process pool."""
        try:
            result = func(item, **kwargs)
            worker_id = f"process_{multiprocessing.current_process().pid}"
            return result, worker_id
        except Exception as e:
            logger.error(f"Process worker failed for {task_id}: {e}")
            raise


class AsyncProcessor(ParallelProcessor):
    """Async-based parallel processor for async operations."""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.max_concurrent = config.max_workers or 10
    
    def process_batch(self, items: List[Any], func: Callable, **kwargs) -> List[ProcessingResult]:
        """Process a batch of items using async concurrency."""
        return asyncio.run(self._async_process_batch(items, func, **kwargs))
    
    def process_stream(self, items: Iterator[Any], func: Callable, **kwargs) -> Iterator[ProcessingResult]:
        """Process a stream of items using async concurrency."""
        return asyncio.run(self._async_process_stream(items, func, **kwargs))
    
    async def _async_process_batch(self, items: List[Any], func: Callable, **kwargs) -> List[ProcessingResult]:
        """Async implementation of batch processing."""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = []
        
        for i, item in enumerate(items):
            task_id = f"async_task_{i}"
            task = asyncio.create_task(
                self._async_safe_execute(semaphore, func, item, task_id, **kwargs)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for i, result in enumerate(results):
            task_id = f"async_task_{i}"
            proc_result = ProcessingResult(task_id)
            
            if isinstance(result, Exception):
                proc_result.fail(result, "async_worker")
            else:
                proc_result.complete(result, "async_worker")
            
            processed_results.append(proc_result)
        
        return processed_results
    
    async def _async_process_stream(self, items: Iterator[Any], func: Callable, **kwargs) -> List[ProcessingResult]:
        """Async implementation of stream processing."""
        # Convert iterator to list for async processing
        item_list = list(items)
        return await self._async_process_batch(item_list, func, **kwargs)
    
    async def _async_safe_execute(self, semaphore: asyncio.Semaphore, func: Callable, 
                                 item: Any, task_id: str, **kwargs) -> Any:
        """Safely execute an async function."""
        async with semaphore:
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(item, **kwargs)
                else:
                    # Run sync function in thread pool
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, partial(func, item, **kwargs))
            except Exception as e:
                logger.error(f"Async execution failed for {task_id}: {e}")
                raise


class ParallelProcessorManager:
    """High-level manager for parallel processing operations."""
    
    def __init__(self, config: Optional[ParallelConfig] = None):
        self.config = config or ParallelConfig()
        self._processors = {
            "thread": ThreadPoolProcessor(self.config),
            "process": ProcessPoolProcessor(self.config),
            "async": AsyncProcessor(self.config)
        }
    
    def process_analysis_data(self, data: List[Any], analysis_func: Callable, 
                            processor_type: str = "thread", **kwargs) -> List[ProcessingResult]:
        """Process analysis data in parallel."""
        if processor_type not in self._processors:
            raise ValueError(f"Unknown processor type: {processor_type}")
        
        processor = self._processors[processor_type]
        logger.info(f"Starting parallel analysis with {processor_type} processor")
        
        start_time = time.time()
        results = processor.process_batch(data, analysis_func, **kwargs)
        duration = time.time() - start_time
        
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        logger.info(f"Parallel analysis completed in {duration:.2f}s: "
                   f"{len(successful_results)} successful, {len(failed_results)} failed")
        
        return results
    
    def process_monitoring_stream(self, data_stream: Iterator[Any], processing_func: Callable,
                                processor_type: str = "thread", **kwargs) -> Iterator[ProcessingResult]:
        """Process monitoring data stream in parallel."""
        if processor_type not in self._processors:
            raise ValueError(f"Unknown processor type: {processor_type}")
        
        processor = self._processors[processor_type]
        logger.info(f"Starting parallel stream processing with {processor_type} processor")
        
        return processor.process_stream(data_stream, processing_func, **kwargs)
    
    def get_optimal_processor_type(self, task_type: str) -> str:
        """Get the optimal processor type for a given task."""
        if task_type in ["analysis", "statistical", "ml"]:
            return "process" if self.config.use_process_pool else "thread"
        elif task_type in ["io", "network", "database"]:
            return "thread"
        elif task_type in ["async", "websocket", "streaming"]:
            return "async"
        else:
            return "thread"  # Default


# Global processor manager
_processor_manager: Optional[ParallelProcessorManager] = None


def get_processor_manager() -> ParallelProcessorManager:
    """Get the global parallel processor manager."""
    global _processor_manager
    if _processor_manager is None:
        _processor_manager = ParallelProcessorManager()
    return _processor_manager


def parallel_analysis(data: List[Any], analysis_func: Callable, **kwargs) -> List[ProcessingResult]:
    """Convenience function for parallel analysis."""
    manager = get_processor_manager()
    processor_type = manager.get_optimal_processor_type("analysis")
    return manager.process_analysis_data(data, analysis_func, processor_type, **kwargs)


def parallel_monitoring(data_stream: Iterator[Any], processing_func: Callable, **kwargs) -> Iterator[ProcessingResult]:
    """Convenience function for parallel monitoring."""
    manager = get_processor_manager()
    processor_type = manager.get_optimal_processor_type("io")
    return manager.process_monitoring_stream(data_stream, processing_func, processor_type, **kwargs)