"""
Comprehensive logging and debugging system for ESCAI CLI.

This module provides structured logging with multiple levels, debug modes,
error tracking, performance logging, and log management capabilities.
"""

import logging
import logging.handlers
import json
import traceback
import time
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from contextlib import contextmanager

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback


class LogLevel(Enum):
    """Enhanced log levels for CLI operations."""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    PERFORMANCE = 25
    USER_ACTION = 15


@dataclass
class LogEntry:
    """Structured log entry with comprehensive metadata."""
    timestamp: str
    level: str
    logger_name: str
    message: str
    module: str
    function: str
    line_number: int
    thread_id: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    command: Optional[str] = None
    execution_time: Optional[float] = None
    memory_usage: Optional[int] = None
    stack_trace: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert log entry to JSON string."""
        return json.dumps(self.to_dict(), default=str, indent=2)


class PerformanceTimer:
    """Context manager for performance timing."""
    
    def __init__(self, logger: 'CLILogger', operation: str, context: Optional[Dict[str, Any]] = None):
        self.logger = logger
        self.operation = operation
        self.context = context or {}
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        self.logger.performance(f"Starting {self.operation}", extra=self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        duration = self.end_time - self.start_time
        
        if exc_type is not None:
            self.logger.error(
                f"Operation {self.operation} failed after {duration:.3f}s",
                extra={**self.context, "duration": duration, "exception": str(exc_val)}
            )
        else:
            self.logger.performance(
                f"Completed {self.operation} in {duration:.3f}s",
                extra={**self.context, "duration": duration}
            )


class CLILogger:
    """Enhanced logger with CLI-specific features."""
    
    def __init__(self, name: str, session_id: Optional[str] = None):
        self.name = name
        self.session_id = session_id
        self.logger = logging.getLogger(name)
        self._context: Dict[str, Any] = {}
        
        # Add custom log levels
        logging.addLevelName(LogLevel.TRACE.value, "TRACE")
        logging.addLevelName(LogLevel.PERFORMANCE.value, "PERFORMANCE")
        logging.addLevelName(LogLevel.USER_ACTION.value, "USER_ACTION")
    
    def set_context(self, **kwargs):
        """Set persistent context for all log messages."""
        self._context.update(kwargs)
    
    def clear_context(self):
        """Clear persistent context."""
        self._context.clear()
    
    def _log_with_context(self, level: int, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log message with enhanced context."""
        combined_extra = {**self._context}
        if extra:
            combined_extra.update(extra)
        
        if self.session_id:
            combined_extra['session_id'] = self.session_id
            
        self.logger.log(level, message, extra=combined_extra)
    
    def trace(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log trace level message."""
        self._log_with_context(LogLevel.TRACE.value, message, extra)
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug level message."""
        self._log_with_context(LogLevel.DEBUG.value, message, extra)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info level message."""
        self._log_with_context(LogLevel.INFO.value, message, extra)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning level message."""
        self._log_with_context(LogLevel.WARNING.value, message, extra)
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None, exc_info: bool = True):
        """Log error level message with optional exception info."""
        if exc_info and sys.exc_info()[0] is not None:
            extra = extra or {}
            extra['stack_trace'] = traceback.format_exc()
        self._log_with_context(LogLevel.ERROR.value, message, extra)
    
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log critical level message."""
        self._log_with_context(LogLevel.CRITICAL.value, message, extra)
    
    def performance(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log performance-related message."""
        self._log_with_context(LogLevel.PERFORMANCE.value, message, extra)
    
    def user_action(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log user action for audit trail."""
        self._log_with_context(LogLevel.USER_ACTION.value, message, extra)
    
    def time_operation(self, operation: str, context: Optional[Dict[str, Any]] = None) -> PerformanceTimer:
        """Create a performance timer context manager."""
        return PerformanceTimer(self, operation, context)


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        # Get caller information
        frame = sys._getframe(8)  # Adjust frame depth as needed
        
        log_entry = LogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=record.levelname,
            logger_name=record.name,
            message=record.getMessage(),
            module=record.module,
            function=record.funcName,
            line_number=record.lineno,
            thread_id=str(threading.get_ident()),
            session_id=getattr(record, 'session_id', None),
            user_id=getattr(record, 'user_id', None),
            command=getattr(record, 'command', None),
            execution_time=getattr(record, 'execution_time', None),
            memory_usage=getattr(record, 'memory_usage', None),
            stack_trace=getattr(record, 'stack_trace', None),
            context=getattr(record, 'context', None)
        )
        
        return log_entry.to_json()


class CLILogManager:
    """Centralized log management system."""
    
    def __init__(self, log_dir: Optional[Path] = None, debug_mode: bool = False):
        self.log_dir = log_dir or Path.home() / ".escai" / "logs"
        self.debug_mode = debug_mode
        self.console = Console()
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Install rich traceback for better error display
        install_rich_traceback(show_locals=debug_mode)
        
        # Configure root logger
        self._setup_root_logger()
        
        # Track active loggers
        self._loggers: Dict[str, CLILogger] = {}
    
    def _setup_root_logger(self):
        """Configure the root logger with appropriate handlers."""
        root_logger = logging.getLogger()
        root_logger.setLevel(LogLevel.TRACE.value if self.debug_mode else LogLevel.INFO.value)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler with Rich formatting
        console_handler = RichHandler(
            console=self.console,
            show_time=True,
            show_level=True,
            show_path=self.debug_mode,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=self.debug_mode
        )
        console_handler.setLevel(LogLevel.INFO.value)
        root_logger.addHandler(console_handler)
        
        # File handler for all logs
        log_file = self.log_dir / f"escai_cli_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(LogLevel.TRACE.value)
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)
        
        # Error file handler for errors and above
        error_file = self.log_dir / f"escai_cli_errors_{datetime.now().strftime('%Y%m%d')}.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(LogLevel.ERROR.value)
        error_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(error_handler)
        
        # Performance log handler
        if self.debug_mode:
            perf_file = self.log_dir / f"escai_cli_performance_{datetime.now().strftime('%Y%m%d')}.log"
            perf_handler = logging.handlers.RotatingFileHandler(
                perf_file,
                maxBytes=5 * 1024 * 1024,  # 5MB
                backupCount=2,
                encoding='utf-8'
            )
            perf_handler.setLevel(LogLevel.PERFORMANCE.value)
            perf_handler.addFilter(lambda record: record.levelno == LogLevel.PERFORMANCE.value)
            perf_handler.setFormatter(StructuredFormatter())
            root_logger.addHandler(perf_handler)
    
    def get_logger(self, name: str, session_id: Optional[str] = None) -> CLILogger:
        """Get or create a logger instance."""
        logger_key = f"{name}:{session_id}" if session_id else name
        
        if logger_key not in self._loggers:
            self._loggers[logger_key] = CLILogger(name, session_id)
        
        return self._loggers[logger_key]
    
    def set_debug_mode(self, enabled: bool):
        """Enable or disable debug mode."""
        self.debug_mode = enabled
        
        # Update log levels
        root_logger = logging.getLogger()
        if enabled:
            root_logger.setLevel(LogLevel.TRACE.value)
            # RichHandler doesn't have show_path attribute in current version
        else:
            root_logger.setLevel(LogLevel.INFO.value)
            # RichHandler doesn't have show_path attribute in current version
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up log files older than specified days."""
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        
        for log_file in self.log_dir.glob("*.log*"):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    self.console.print(f"[dim]Cleaned up old log file: {log_file.name}[/dim]")
                except OSError as e:
                    self.console.print(f"[yellow]Warning: Could not delete {log_file.name}: {e}[/yellow]")
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get statistics about log files."""
        stats = {
            "log_directory": str(self.log_dir),
            "total_files": 0,
            "total_size_mb": 0,
            "files": []
        }
        
        for log_file in self.log_dir.glob("*.log*"):
            file_size = log_file.stat().st_size
            files_list: List[Dict[str, Any]] = stats["files"]  # type: ignore[assignment]
            files_list.append({
                "name": log_file.name,
                "size_mb": round(file_size / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
            })
            total_size: float = stats["total_size_mb"]  # type: ignore[assignment]
            total_files: int = stats["total_files"]  # type: ignore[assignment]
            stats["total_size_mb"] = total_size + (file_size / (1024 * 1024))
            stats["total_files"] = total_files + 1
        
        stats["total_size_mb"] = round(float(stats["total_size_mb"]), 2)  # type: ignore[arg-type]
        return stats


# Global log manager instance
_log_manager: Optional[CLILogManager] = None


def initialize_logging(log_dir: Optional[Path] = None, debug_mode: bool = False) -> CLILogManager:
    """Initialize the global logging system."""
    global _log_manager
    _log_manager = CLILogManager(log_dir, debug_mode)
    return _log_manager


def get_logger(name: str, session_id: Optional[str] = None) -> CLILogger:
    """Get a logger instance from the global manager."""
    if _log_manager is None:
        initialize_logging()
    return _log_manager.get_logger(name, session_id)


def set_debug_mode(enabled: bool):
    """Set debug mode globally."""
    if _log_manager is None:
        initialize_logging()
    _log_manager.set_debug_mode(enabled)


@contextmanager
def log_context(**kwargs):
    """Context manager for temporary logging context."""
    logger = get_logger("context")
    logger.set_context(**kwargs)
    try:
        yield logger
    finally:
        logger.clear_context()