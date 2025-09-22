"""
Progress indicators with accurate time estimation for CLI operations.
Implements various progress display formats with intelligent time prediction.
"""

import time
import threading
from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field
from collections import deque
import math
import statistics
from rich.console import Console
from rich.progress import (
    Progress, TaskID, BarColumn, TextColumn, TimeRemainingColumn,
    TimeElapsedColumn, SpinnerColumn, MofNCompleteColumn
)
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProgressConfig:
    """Configuration for progress indicators."""
    update_interval: float = 0.1  # Update interval in seconds
    time_window_size: int = 50  # Number of samples for time estimation
    smoothing_factor: float = 0.3  # Exponential smoothing factor
    show_percentage: bool = True
    show_eta: bool = True
    show_speed: bool = True
    show_elapsed: bool = True
    auto_refresh: bool = True
    console_width: Optional[int] = None


class TimeEstimator:
    """Intelligent time estimation for progress tracking."""
    
    def __init__(self, window_size: int = 50, smoothing_factor: float = 0.3):
        self.window_size = window_size
        self.smoothing_factor = smoothing_factor
        self.timestamps: deque = deque(maxlen=window_size)
        self.progress_values: deque = deque(maxlen=window_size)
        self.start_time = time.time()
        self.last_update = self.start_time
        self.smoothed_rate = 0.0
        self.rate_history: deque = deque(maxlen=10)
    
    def update(self, current_progress: float, total_progress: float):
        """Update the estimator with new progress data."""
        current_time = time.time()
        
        # Store timestamp and progress
        self.timestamps.append(current_time)
        self.progress_values.append(current_progress)
        
        # Calculate instantaneous rate
        if len(self.timestamps) >= 2:
            time_diff = current_time - self.timestamps[-2]
            progress_diff = current_progress - self.progress_values[-2]
            
            if time_diff > 0:
                instantaneous_rate = progress_diff / time_diff
                self.rate_history.append(instantaneous_rate)
                
                # Update smoothed rate
                if self.smoothed_rate == 0:
                    self.smoothed_rate = instantaneous_rate
                else:
                    self.smoothed_rate = (self.smoothing_factor * instantaneous_rate + 
                                        (1 - self.smoothing_factor) * self.smoothed_rate)
        
        self.last_update = current_time
    
    def get_eta_seconds(self, current_progress: float, total_progress: float) -> Optional[float]:
        """Get estimated time to completion in seconds."""
        if current_progress >= total_progress or self.smoothed_rate <= 0:
            return None
        
        remaining_progress = total_progress - current_progress
        
        # Use multiple estimation methods and take the median
        estimates = []
        
        # Method 1: Smoothed rate
        if self.smoothed_rate > 0:
            estimates.append(remaining_progress / self.smoothed_rate)
        
        # Method 2: Linear regression on recent data
        if len(self.timestamps) >= 5:
            recent_eta = self._linear_regression_eta(current_progress, total_progress)
            if recent_eta:
                estimates.append(recent_eta)
        
        # Method 3: Average rate over time window
        if len(self.rate_history) >= 3:
            avg_rate = statistics.mean(self.rate_history)
            if avg_rate > 0:
                estimates.append(remaining_progress / avg_rate)
        
        # Return median estimate if we have multiple estimates
        if estimates:
            return statistics.median(estimates)
        
        return None
    
    def _linear_regression_eta(self, current_progress: float, total_progress: float) -> Optional[float]:
        """Calculate ETA using linear regression on recent progress."""
        if len(self.timestamps) < 5:
            return None
        
        # Use recent data points
        recent_times = list(self.timestamps)[-10:]
        recent_progress = list(self.progress_values)[-10:]
        
        if len(recent_times) < 3:
            return None
        
        # Simple linear regression
        n = len(recent_times)
        sum_t = sum(recent_times)
        sum_p = sum(recent_progress)
        sum_tp = sum(t * p for t, p in zip(recent_times, recent_progress))
        sum_t2 = sum(t * t for t in recent_times)
        
        denominator = n * sum_t2 - sum_t * sum_t
        if abs(denominator) < 1e-10:
            return None
        
        # Calculate slope (rate)
        slope = (n * sum_tp - sum_t * sum_p) / denominator
        
        if slope <= 0:
            return None
        
        # Estimate time to reach total_progress
        current_time = time.time()
        remaining_progress = total_progress - current_progress
        
        return remaining_progress / slope
    
    def get_speed(self) -> float:
        """Get current processing speed (items per second)."""
        return max(0, self.smoothed_rate)
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since start."""
        return time.time() - self.start_time


class ProgressIndicator:
    """Base class for progress indicators."""
    
    def __init__(self, total: int, description: str = "", config: Optional[ProgressConfig] = None):
        self.total = total
        self.description = description
        self.config = config or ProgressConfig()
        self.current = 0
        self.estimator = TimeEstimator(
            self.config.time_window_size,
            self.config.smoothing_factor
        )
        self._lock = threading.Lock()
        self._callbacks: List[Callable] = []
    
    def update(self, advance: int = 1):
        """Update progress by advancing the counter."""
        with self._lock:
            self.current = min(self.current + advance, self.total)
            self.estimator.update(self.current, self.total)
            
            # Call callbacks
            for callback in self._callbacks:
                try:
                    callback(self.get_progress_info())
                except Exception as e:
                    logger.error(f"Progress callback error: {e}")
    
    def set_progress(self, value: int):
        """Set absolute progress value."""
        with self._lock:
            self.current = min(max(0, value), self.total)
            self.estimator.update(self.current, self.total)
    
    def add_callback(self, callback: Callable):
        """Add a progress callback function."""
        self._callbacks.append(callback)
    
    def get_progress_info(self) -> Dict[str, Any]:
        """Get comprehensive progress information."""
        with self._lock:
            percentage = (self.current / self.total * 100) if self.total > 0 else 0
            eta_seconds = self.estimator.get_eta_seconds(self.current, self.total)
            
            return {
                "current": self.current,
                "total": self.total,
                "percentage": percentage,
                "description": self.description,
                "elapsed_seconds": self.estimator.get_elapsed_time(),
                "eta_seconds": eta_seconds,
                "speed": self.estimator.get_speed(),
                "is_complete": self.current >= self.total
            }
    
    def is_complete(self) -> bool:
        """Check if progress is complete."""
        return self.current >= self.total


class RichProgressIndicator(ProgressIndicator):
    """Rich-based progress indicator with advanced formatting."""
    
    def __init__(self, total: int, description: str = "", config: Optional[ProgressConfig] = None):
        super().__init__(total, description, config)
        self.console = Console(width=config.console_width if config else None)
        self.progress = None
        self.task_id = None
        self._live = None
        self._running = False
    
    def start(self):
        """Start the progress display."""
        if self._running:
            return
        
        # Create progress bar with custom columns
        columns = [
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
        ]
        
        if self.config.show_percentage:
            columns.append(TextColumn("[progress.percentage]{task.percentage:>3.0f}%"))
        
        if self.config.show_speed:
            columns.append(TextColumn("[cyan]{task.speed:.1f} items/s"))
        
        if self.config.show_elapsed:
            columns.append(TimeElapsedColumn())
        
        if self.config.show_eta:
            columns.append(TimeRemainingColumn())
        
        self.progress = Progress(*columns, console=self.console, auto_refresh=self.config.auto_refresh)
        self.task_id = self.progress.add_task(self.description, total=self.total)
        
        self._live = Live(self.progress, console=self.console, refresh_per_second=1/self.config.update_interval)
        self._live.start()
        self._running = True
    
    def stop(self):
        """Stop the progress display."""
        if not self._running:
            return
        
        if self._live:
            self._live.stop()
        self._running = False
    
    def update(self, advance: int = 1):
        """Update progress and display."""
        super().update(advance)
        
        if self.progress and self.task_id is not None:
            # Update Rich progress
            self.progress.update(
                self.task_id,
                completed=self.current,
                speed=self.estimator.get_speed()
            )
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class MultiProgressIndicator:
    """Manages multiple progress indicators simultaneously."""
    
    def __init__(self, config: Optional[ProgressConfig] = None):
        self.config = config or ProgressConfig()
        self.console = Console(width=config.console_width if config else None)
        self.progress_bars: Dict[str, ProgressIndicator] = {}
        self.progress = None
        self.task_ids: Dict[str, TaskID] = {}
        self._live = None
        self._running = False
        self._lock = threading.Lock()
    
    def add_progress(self, name: str, total: int, description: str = "") -> ProgressIndicator:
        """Add a new progress indicator."""
        with self._lock:
            indicator = ProgressIndicator(total, description or name, self.config)
            self.progress_bars[name] = indicator
            
            if self._running and self.progress:
                task_id = self.progress.add_task(description or name, total=total)
                self.task_ids[name] = task_id
            
            return indicator
    
    def update_progress(self, name: str, advance: int = 1):
        """Update a specific progress indicator."""
        with self._lock:
            if name in self.progress_bars:
                self.progress_bars[name].update(advance)
                
                if self._running and self.progress and name in self.task_ids:
                    current = self.progress_bars[name].current
                    speed = self.progress_bars[name].estimator.get_speed()
                    self.progress.update(self.task_ids[name], completed=current, speed=speed)
    
    def start(self):
        """Start all progress displays."""
        if self._running:
            return
        
        # Create progress with columns
        columns = [
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[cyan]{task.speed:.1f}/s"),
            TimeRemainingColumn(),
        ]
        
        self.progress = Progress(*columns, console=self.console)
        
        # Add existing progress bars
        with self._lock:
            for name, indicator in self.progress_bars.items():
                task_id = self.progress.add_task(
                    indicator.description or name,
                    total=indicator.total,
                    completed=indicator.current
                )
                self.task_ids[name] = task_id
        
        self._live = Live(self.progress, console=self.console, refresh_per_second=10)
        self._live.start()
        self._running = True
    
    def stop(self):
        """Stop all progress displays."""
        if not self._running:
            return
        
        if self._live:
            self._live.stop()
        self._running = False
    
    def get_overall_progress(self) -> Dict[str, Any]:
        """Get overall progress across all indicators."""
        with self._lock:
            if not self.progress_bars:
                return {"percentage": 0, "completed": 0, "total": 0}
            
            total_completed = sum(p.current for p in self.progress_bars.values())
            total_items = sum(p.total for p in self.progress_bars.values())
            
            percentage = (total_completed / total_items * 100) if total_items > 0 else 0
            
            return {
                "percentage": percentage,
                "completed": total_completed,
                "total": total_items,
                "active_tasks": len(self.progress_bars),
                "completed_tasks": sum(1 for p in self.progress_bars.values() if p.is_complete())
            }
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class AnalysisProgressTracker:
    """Specialized progress tracker for analysis operations."""
    
    def __init__(self, total_items: int, analysis_name: str = "Analysis"):
        self.total_items = total_items
        self.analysis_name = analysis_name
        self.multi_progress = MultiProgressIndicator()
        
        # Create progress bars for different phases
        self.phases = {
            "loading": self.multi_progress.add_progress("loading", total_items, "Loading data"),
            "processing": self.multi_progress.add_progress("processing", total_items, "Processing"),
            "analysis": self.multi_progress.add_progress("analysis", total_items, "Analyzing"),
            "output": self.multi_progress.add_progress("output", 1, "Generating output")
        }
        
        self.current_phase = "loading"
    
    def start_phase(self, phase_name: str):
        """Start a specific analysis phase."""
        if phase_name in self.phases:
            self.current_phase = phase_name
    
    def update_current_phase(self, advance: int = 1):
        """Update the current phase progress."""
        if self.current_phase in self.phases:
            self.multi_progress.update_progress(self.current_phase, advance)
    
    def complete_phase(self, phase_name: str):
        """Mark a phase as complete."""
        if phase_name in self.phases:
            remaining = self.phases[phase_name].total - self.phases[phase_name].current
            if remaining > 0:
                self.multi_progress.update_progress(phase_name, remaining)
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of analysis progress."""
        overall = self.multi_progress.get_overall_progress()
        phase_status = {}
        
        for phase_name, indicator in self.phases.items():
            info = indicator.get_progress_info()
            phase_status[phase_name] = {
                "percentage": info["percentage"],
                "complete": info["is_complete"]
            }
        
        return {
            "analysis_name": self.analysis_name,
            "overall_progress": overall,
            "phases": phase_status
        }
    
    def __enter__(self):
        self.multi_progress.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.multi_progress.stop()


# Convenience functions
def create_progress_bar(total: int, description: str = "", config: Optional[ProgressConfig] = None) -> RichProgressIndicator:
    """Create a rich progress bar."""
    return RichProgressIndicator(total, description, config)


def create_multi_progress(config: Optional[ProgressConfig] = None) -> MultiProgressIndicator:
    """Create a multi-progress indicator."""
    return MultiProgressIndicator(config)


def create_analysis_tracker(total_items: int, analysis_name: str = "Analysis") -> AnalysisProgressTracker:
    """Create an analysis progress tracker."""
    return AnalysisProgressTracker(total_items, analysis_name)